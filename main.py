import os
import cv2
import time
import random
import pickle
import numpy as np
from PIL import Image
from distutils.version import LooseVersion

from sacred import Experiment
from easydict import EasyDict as edict

import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as tf

from models.baseline_same import Baseline as UNet
from utils.loss import hinge_embedding_loss, surface_normal_loss, parameter_loss, \
    class_balanced_cross_entropy_loss
from utils.misc import AverageMeter, get_optimizer
from utils.metric import eval_iou, eval_plane_prediction
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors
from bin_mean_shift import Bin_Mean_Shift
from modules import get_coordinate_map
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss
from match_segmentation import MatchSegmentation

ex = Experiment()


class PlaneDataset(data.Dataset):
    def __init__(self, subset='train', transform=None, root_dir=None):
        assert subset in ['train', 'val']
        self.subset = subset
        self.transform = transform
        self.root_dir = os.path.join(root_dir, subset)
        self.txt_file = os.path.join(root_dir, subset + '.txt')

        self.data_list = [line.strip() for line in open(self.txt_file, 'r').readlines()]
        self.precompute_K_inv_dot_xy_1()

    def get_plane_parameters(self, plane, plane_nums, segmentation):
        valid_region = segmentation != 20

        plane = plane[:plane_nums]

        tmp = plane[:, 1].copy()
        plane[:, 1] = -plane[:, 2]
        plane[:, 2] = tmp

        # convert plane from n * d to n / d
        plane_d = np.linalg.norm(plane, axis=1)
        # normalize
        plane /= plane_d.reshape(-1, 1)
        # n / d
        plane /= plane_d.reshape(-1, 1)

        h, w = segmentation.shape
        plane_parameters = np.ones((3, h, w))
        for i in range(h):
            for j in range(w):
                d = segmentation[i, j]
                if d >= 20: continue
                plane_parameters[:, i, j] = plane[d, :]

        # plane_instance parameter, padding zero to fix size
        plane_instance_parameter = np.concatenate((plane, np.zeros((20-plane.shape[0], 3))), axis=0)
        return plane_parameters, valid_region, plane_instance_parameter

    def precompute_K_inv_dot_xy_1(self, h=192, w=256):
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x],
             [0, focal_length, offset_y],
             [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))
        self.K_inv = K_inv

        K_inv_dot_xy_1 = np.zeros((3, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640
                
                ray = np.dot(self.K_inv,
                             np.array([xx, yy, 1]).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]

        # precompute to speed up processing
        self.K_inv_dot_xy_1 = K_inv_dot_xy_1

    def plane2depth(self, plane_parameters, num_planes, segmentation, gt_depth, h=192, w=256):
            
        depth_map = 1. / np.sum(self.K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
        depth_map = depth_map.reshape(h, w)

        # replace non planer region depth using sensor depth map
        depth_map[segmentation == 20] = gt_depth[segmentation == 20]
        return depth_map

    def __getitem__(self, index):
        if self.subset == 'train':
            data_path = self.data_list[index]
        else:
            data_path = str(index) + '.npz'
        data_path = os.path.join(self.root_dir, data_path)
        data = np.load(data_path)

        image = data['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        plane = data['plane']
        num_planes = data['num_planes'][0]

        gt_segmentation = data['segmentation']
        gt_segmentation = gt_segmentation.reshape((192, 256))
        segmentation = np.zeros([21, 192, 256], dtype=np.uint8)

        _, h, w = segmentation.shape
        for i in range(num_planes+1):
            # deal with backgroud
            if i == num_planes:
                seg = gt_segmentation == 20
            else:
                seg = gt_segmentation == i

            segmentation[i, :, :] = seg.reshape(h, w)

        # surface plane parameters
        plane_parameters, valid_region, plane_instance_parameter = \
            self.get_plane_parameters(plane, num_planes, gt_segmentation)

        # since some depth is missing, we use plane to recover those depth following PlaneNet
        gt_depth = data['depth'].reshape(192, 256)
        depth = self.plane2depth(plane_parameters, num_planes, gt_segmentation, gt_depth).reshape(1, 192, 256)

        sample = {
            'image': image,
            'num_planes': num_planes,
            'instance': torch.ByteTensor(segmentation),
            # one for planar and zero for non-planar
            'semantic': 1 - torch.FloatTensor(segmentation[num_planes, :, :]).unsqueeze(0),
            'gt_seg': torch.LongTensor(gt_segmentation),
            'depth': torch.FloatTensor(depth),
            'plane_parameters': torch.FloatTensor(plane_parameters),
            'valid_region': torch.ByteTensor(valid_region.astype(np.uint8)).unsqueeze(0),
            'plane_instance_parameter': torch.FloatTensor(plane_instance_parameter)
        }

        return sample

    def __len__(self):
        return len(self.data_list)


def load_dataset(subset, cfg):
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    is_shuffle = subset == 'train'
    loaders = data.DataLoader(
        PlaneDataset(subset=subset, transform=transforms, root_dir=cfg.root_dir),
        batch_size=cfg.batch_size, shuffle=is_shuffle, num_workers=cfg.num_workers
    )

    return loaders


@ex.command
def train(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not (_run._id is None):
        checkpoint_dir = os.path.join(_run.observers[0].basedir, str(_run._id), 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # build network
    network = UNet(cfg.model)

    if not (cfg.resume_dir == 'None'):
        model_dict = torch.load(cfg.resume_dir, map_location=lambda storage, loc: storage)
        network.load_state_dict(model_dict)

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)

    # set up optimizers
    optimizer = get_optimizer(network.parameters(), cfg.solver)

    # data loader
    data_loader = load_dataset('train', cfg.dataset)

    # save losses per epoch
    history = {'losses': [], 'losses_pull': [], 'losses_push': [],
               'losses_binary': [], 'losses_depth': [], 'ioues': [], 'rmses': []}

    network.train(not cfg.model.fix_bn)

    bin_mean_shift = Bin_Mean_Shift(device=device)
    k_inv_dot_xy1 = get_coordinate_map(device)
    instance_parameter_loss = InstanceParameterLoss(k_inv_dot_xy1)

    # main loop
    for epoch in range(cfg.num_epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_pull = AverageMeter()
        losses_push = AverageMeter()
        losses_binary = AverageMeter()
        losses_depth = AverageMeter()
        losses_normal = AverageMeter()
        losses_instance = AverageMeter()
        ioues = AverageMeter()
        rmses = AverageMeter()
        instance_rmses = AverageMeter()
        mean_angles = AverageMeter()

        tic = time.time()
        for iter, sample in enumerate(data_loader):
            image = sample['image'].to(device)
            instance = sample['instance'].to(device)
            semantic = sample['semantic'].to(device)
            gt_depth = sample['depth'].to(device)
            gt_seg = sample['gt_seg'].to(device)
            gt_plane_parameters = sample['plane_parameters'].to(device)
            valid_region = sample['valid_region'].to(device)
            gt_plane_instance_parameter = sample['plane_instance_parameter'].to(device)

            # forward pass
            logit, embedding, _, _, param = network(image)

            segmentations, sample_segmentations, sample_params, centers, sample_probs, sample_gt_segs = \
                bin_mean_shift(logit, embedding, param, gt_seg)

            # calculate loss
            loss, loss_pull, loss_push, loss_binary, loss_depth, loss_normal, loss_parameters, loss_pw, loss_instance \
                = 0., 0., 0., 0., 0., 0., 0., 0., 0.
            batch_size = image.size(0)
            for i in range(batch_size):
                _loss, _loss_pull, _loss_push = hinge_embedding_loss(embedding[i:i+1], sample['num_planes'][i:i+1],
                                                                     instance[i:i+1], device)

                _loss_binary = class_balanced_cross_entropy_loss(logit[i], semantic[i])

                _loss_normal, mean_angle = surface_normal_loss(param[i:i+1], gt_plane_parameters[i:i+1],
                                                               valid_region[i:i+1])

                _loss_L1 = parameter_loss(param[i:i + 1], gt_plane_parameters[i:i + 1], valid_region[i:i + 1])
                _loss_depth, rmse, infered_depth = Q_loss(param[i:i+1], k_inv_dot_xy1, gt_depth[i:i+1])

                if segmentations[i] is None:
                    continue

                _instance_loss, instance_depth, instance_abs_disntace, _ = \
                    instance_parameter_loss(segmentations[i], sample_segmentations[i], sample_params[i],
                                            valid_region[i:i+1], gt_depth[i:i+1])

                _loss += _loss_binary + _loss_depth + _loss_normal + _instance_loss + _loss_L1

                # planar segmentation iou
                prob = torch.sigmoid(logit[i])
                mask = (prob > 0.5).float().cpu().numpy()
                iou = eval_iou(mask, semantic[i].cpu().numpy())
                ioues.update(iou * 100)
                instance_rmses.update(instance_abs_disntace.item())
                rmses.update(rmse.item())
                mean_angles.update(mean_angle.item())

                loss += _loss
                loss_pull += _loss_pull
                loss_push += _loss_push
                loss_binary += _loss_binary
                loss_depth += _loss_depth
                loss_normal += _loss_normal
                loss_instance += _instance_loss

            loss /= batch_size
            loss_pull /= batch_size
            loss_push /= batch_size
            loss_binary /= batch_size
            loss_depth /= batch_size
            loss_normal /= batch_size
            loss_instance /= batch_size

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss
            losses.update(loss.item())
            losses_pull.update(loss_pull.item())
            losses_push.update(loss_push.item())
            losses_binary.update(loss_binary.item())
            losses_depth.update(loss_depth.item())
            losses_normal.update(loss_normal.item())
            losses_instance.update(loss_instance.item())

            # update time
            batch_time.update(time.time() - tic)
            tic = time.time()

            if iter % cfg.print_interval == 0:
                _log.info(f"[{epoch:2d}][{iter:5d}/{len(data_loader):5d}] "
                          f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                          f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                          f"Pull: {losses_pull.val:.4f} ({losses_pull.avg:.4f}) "
                          f"Push: {losses_push.val:.4f} ({losses_push.avg:.4f}) "
                          f"INS: {losses_instance.val:.4f} ({losses_instance.avg:.4f}) "
                          f"Binary: {losses_binary.val:.4f} ({losses_binary.avg:.4f}) "
                          f"IoU: {ioues.val:.2f} ({ioues.avg:.2f}) "
                          f"LN: {losses_normal.val:.4f} ({losses_normal.avg:.4f}) "
                          f"AN: {mean_angles.val:.4f} ({mean_angles.avg:.4f}) "
                          f"Depth: {losses_depth.val:.4f} ({losses_depth.avg:.4f}) "
                          f"INSDEPTH: {instance_rmses.val:.4f} ({instance_rmses.avg:.4f}) "
                          f"RMSE: {rmses.val:.4f} ({rmses.avg:.4f}) ")

        _log.info(f"* epoch: {epoch:2d}\t"
                  f"Loss: {losses.avg:.6f}\t"
                  f"Pull: {losses_pull.avg:.6f}\t"
                  f"Push: {losses_push.avg:.6f}\t"
                  f"Binary: {losses_binary.avg:.6f}\t"
                  f"Depth: {losses_depth.avg:.6f}\t"
                  f"IoU: {ioues.avg:.2f}\t"
                  f"RMSE: {rmses.avg:.4f}\t")

        # save history
        history['losses'].append(losses.avg)
        history['losses_pull'].append(losses_pull.avg)
        history['losses_push'].append(losses_push.avg)
        history['losses_binary'].append(losses_binary.avg)
        history['losses_depth'].append(losses_depth.avg)
        history['ioues'].append(ioues.avg)
        history['rmses'].append(rmses.avg)

        # save checkpoint
        if not (_run._id is None):
            torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))
            pickle.dump(history, open(os.path.join(checkpoint_dir, 'history.pkl'), 'wb'))


@ex.command
def eval(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not (_run._id is None):
        checkpoint_dir = os.path.join('experiments', str(_run._id), 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # build network
    network = UNet(cfg.model)

    if not (cfg.resume_dir == 'None'):
        model_dict = torch.load(cfg.resume_dir, map_location=lambda storage, loc: storage)
        network.load_state_dict(model_dict)

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)
    network.eval()

    # data loader
    data_loader = load_dataset('val', cfg.dataset)

    pixel_recall_curve = np.zeros((13))
    plane_recall_curve = np.zeros((13, 3))

    bin_mean_shift = Bin_Mean_Shift(device=device)
    k_inv_dot_xy1 = get_coordinate_map(device)
    instance_parameter_loss = InstanceParameterLoss(k_inv_dot_xy1)
    match_segmentatin = MatchSegmentation()

    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            image = sample['image'].to(device)
            instance = sample['instance'].to(device)
            gt_seg = sample['gt_seg'].numpy()
            semantic = sample['semantic'].to(device)
            gt_depth = sample['depth'].to(device)
            # gt_plane_parameters = sample['plane_parameters'].to(device)
            valid_region = sample['valid_region'].to(device)
            gt_plane_num = sample['num_planes'].int()
            # gt_plane_instance_parameter = sample['plane_instance_parameter'].numpy()
            
            # forward pass
            logit, embedding, _, _, param = network(image)

            prob = torch.sigmoid(logit[0])
            
            # infer per pixel depth using per pixel plane parameter
            _, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, gt_depth)

            # fast mean shift
            segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(
                prob, embedding[0], param, mask_threshold=0.1)

            # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
            # we thus use avg_pool_2d to smooth the segmentation results
            b = segmentation.t().view(1, -1, 192, 256)
            pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
            b = pooling_b.view(-1, 192*256).t()
            segmentation = b

            # infer instance depth
            instance_loss, instance_depth, instance_abs_disntace, instance_parameter = \
                instance_parameter_loss(segmentation, sampled_segmentation, sample_param,
                                        valid_region, gt_depth, False)

            # greedy match of predict segmentation and ground truth segmentation using cross entropy
            # to better visualization
            matching = match_segmentatin(segmentation, prob.view(-1, 1), instance[0], gt_plane_num)

            # return cluster results
            predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

            # reindexing to matching gt segmentation for better visualization
            matching = matching.cpu().numpy().reshape(-1)
            used = set([])
            max_index = max(matching) + 1
            for i, a in zip(range(len(matching)), matching):
                if a in used:
                    matching[i] = max_index
                    max_index += 1
                else:
                    used.add(a)
            predict_segmentation = matching[predict_segmentation]

            # mask out non planar region
            predict_segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
            predict_segmentation = predict_segmentation.reshape(192, 256)

            # visualization and evaluation
            h, w = 192, 256
            image = tensor_to_image(image.cpu()[0])
            semantic = semantic.cpu().numpy().reshape(h, w)
            mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)
            gt_seg = gt_seg.reshape(h, w)
            depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
            per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

            # use per pixel depth for non planar region
            depth = depth * (predict_segmentation != 20) + per_pixel_depth * (predict_segmentation == 20)
            gt_depth = gt_depth.cpu().numpy()[0, 0].reshape(h, w)

            # evaluation plane segmentation
            pixelStatistics, planeStatistics = eval_plane_prediction(
                predict_segmentation, gt_seg, depth, gt_depth)

            pixel_recall_curve += np.array(pixelStatistics)
            plane_recall_curve += np.array(planeStatistics)

            print("pixel and plane recall of test image ", iter)
            print(pixel_recall_curve / float(iter+1))
            print(plane_recall_curve[:, 0] / plane_recall_curve[:, 1])
            print("********")

            # visualization convert labels to color image
            # change non-planar regions to zero, so non-planar regions use the black color
            gt_seg += 1
            gt_seg[gt_seg == 21] = 0
            predict_segmentation += 1
            predict_segmentation[predict_segmentation == 21] = 0

            gt_seg_image = cv2.resize(np.stack([colors[gt_seg, 0],
                                                colors[gt_seg, 1],
                                                colors[gt_seg, 2]], axis=2), (w, h))
            pred_seg = cv2.resize(np.stack([colors[predict_segmentation, 0],
                                            colors[predict_segmentation, 1],
                                            colors[predict_segmentation, 2]], axis=2), (w, h))

            # blend image
            blend_pred = (pred_seg * 0.7 + image * 0.3).astype(np.uint8)
            blend_gt = (gt_seg_image * 0.7 + image * 0.3).astype(np.uint8)

            semantic = cv2.resize((semantic * 255).astype(np.uint8), (w, h))
            semantic = cv2.cvtColor(semantic, cv2.COLOR_GRAY2BGR)

            mask = cv2.resize((mask * 255).astype(np.uint8), (w, h))
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            depth_diff = np.abs(gt_depth - depth)
            depth_diff[gt_depth == 0.] = 0

            # visualize depth map as PlaneNet
            depth_diff = np.clip(depth_diff / 5 * 255, 0, 255).astype(np.uint8)
            depth_diff = cv2.cvtColor(cv2.resize(depth_diff, (w, h)), cv2.COLOR_GRAY2BGR)

            depth = 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
            depth = cv2.cvtColor(cv2.resize(depth, (w, h)), cv2.COLOR_GRAY2BGR)

            gt_depth = 255 - np.clip(gt_depth / 5 * 255, 0, 255).astype(np.uint8)
            gt_depth = cv2.cvtColor(cv2.resize(gt_depth, (w, h)), cv2.COLOR_GRAY2BGR)

            image_1 = np.concatenate((image, pred_seg, gt_seg_image), axis=1)
            image_2 = np.concatenate((image, blend_pred, blend_gt), axis=1)
            image_3 = np.concatenate((image, mask, semantic), axis=1)
            image_4 = np.concatenate((depth_diff, depth, gt_depth), axis=1)
            image = np.concatenate((image_1, image_2, image_3, image_4), axis=0)

            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            # cv2.imwrite("%d_segmentation.png"%iter, image)

        print("========================================")
        print("pixel and plane recall of all test image")
        print(pixel_recall_curve / len(data_loader))
        print(plane_recall_curve[:, 0] / plane_recall_curve[:, 1])
        print("****************************************")


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    ex.add_config('./configs/config.yaml')
    ex.run_commandline()
