import numpy as np
import torch
import torch.nn.functional as F


# https://github.com/kmaninis/OSVOS-PyTorch
def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = label.float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()

    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= int(np.prod(label.size()))
    elif batch_average:
        final_loss /= int(label.size(0))

    return final_loss


def hinge_embedding_loss(embedding, num_planes, segmentation, device, t_pull=0.5, t_push=1.5):
    b, c, h, w = embedding.size()
    assert(b == 1)

    num_planes = num_planes.numpy()[0]
    embedding = embedding[0]
    segmentation = segmentation[0]
    embeddings = []
    # select embedding with segmentation
    for i in range(num_planes):
        feature = torch.transpose(torch.masked_select(embedding, segmentation[i, :, :].view(1, h, w)).view(c, -1), 0, 1)
        embeddings.append(feature)

    centers = []
    for feature in embeddings:
        center = torch.mean(feature, dim=0).view(1, c)
        centers.append(center)

    # intra-embedding loss within a plane
    pull_loss = torch.Tensor([0.0]).to(device)
    for feature, center in zip(embeddings, centers):
        dis = torch.norm(feature - center, 2, dim=1) - t_pull
        dis = F.relu(dis)
        pull_loss += torch.mean(dis)
    pull_loss /= int(num_planes)

    if num_planes == 1:
        return pull_loss, pull_loss, torch.zeros(1).to(device)

    # inter-plane loss
    centers = torch.cat(centers, dim=0)
    A = centers.repeat(1, int(num_planes)).view(-1, c)
    B = centers.repeat(int(num_planes), 1)
    distance = torch.norm(A - B, 2, dim=1).view(int(num_planes), int(num_planes))

    # select pair wise distance from distance matrix
    eye = torch.eye(int(num_planes)).to(device)
    pair_distance = torch.masked_select(distance, eye == 0)

    pair_distance = t_push - pair_distance
    pair_distance = F.relu(pair_distance)
    push_loss = torch.mean(pair_distance).view(-1)

    loss = pull_loss + push_loss
    return loss, pull_loss, push_loss


def surface_normal_loss(prediction, surface_normal, valid_region):
    b, c, h, w = prediction.size()
    if valid_region is None:
        valid_predition = torch.transpose(prediction.view(c, -1), 0, 1)
        valid_surface_normal = torch.transpose(surface_normal.view(c, -1), 0, 1)
    else:
        valid_predition = torch.transpose(torch.masked_select(prediction, valid_region).view(c, -1), 0, 1)
        valid_surface_normal = torch.transpose(torch.masked_select(surface_normal, valid_region).view(c, -1), 0, 1)

    similarity = torch.nn.functional.cosine_similarity(valid_predition, valid_surface_normal, dim=1)

    loss = torch.mean(1-similarity)
    mean_angle = torch.mean(torch.acos(torch.clamp(similarity, -1, 1)))
    return loss, mean_angle / np.pi * 180


# L1 parameter loss
def parameter_loss(prediction, param, valid_region):
    b, c, h, w = prediction.size()
    if valid_region is None:
        valid_predition = torch.transpose(prediction.view(c, -1), 0, 1)
        valid_param = torch.transpose(param.view(c, -1), 0, 1)
    else:
        valid_predition = torch.transpose(torch.masked_select(prediction, valid_region).view(c, -1), 0, 1)
        valid_param = torch.transpose(torch.masked_select(param, valid_region).view(c, -1), 0, 1)

    return torch.mean(torch.sum(torch.abs(valid_predition - valid_param), dim=1))


def Q_loss(param, k_inv_dot_xy1, gt_depth):
    '''
    infer per pixel depth using perpixel plane parameter and
    return depth loss, mean abs distance to gt depth, perpixel depth map
    :param param: plane parameters defined as n/d , tensor with size (1, 3, h, w)
    :param k_inv_dot_xy1: tensor with size (3, h*w)
    :param depth: tensor with size(1, 1, h, w)
    :return: error and abs distance
    '''

    b, c, h, w = param.size()
    assert (b == 1 and c == 3)

    gt_depth = gt_depth.view(1, h*w)
    param = param.view(c, h*w)

    # infer depth for every pixel
    infered_depth = 1. / torch.sum(param * k_inv_dot_xy1, dim=0, keepdim=True)  # (1, h*w)
    infered_depth = infered_depth.view(1, h * w)

    # ignore insufficient depth
    infered_depth = torch.clamp(infered_depth, 1e-4, 10.0)

    # select valid depth
    mask = gt_depth != 0.0
    valid_gt_depth = torch.masked_select(gt_depth, mask)
    valid_depth = torch.masked_select(infered_depth, mask)
    valid_param = torch.masked_select(param, mask).view(3, -1)
    valid_ray = torch.masked_select(k_inv_dot_xy1, mask).view(3, -1)

    diff = torch.abs(valid_depth - valid_gt_depth)
    abs_distance = torch.mean(diff)

    Q = valid_ray * valid_gt_depth   # (3, n)
    q_diff = torch.abs(torch.sum(valid_param * Q, dim=0, keepdim=True) - 1.)
    loss = torch.mean(q_diff)
    return loss, abs_distance, infered_depth.view(1, 1, h, w)
