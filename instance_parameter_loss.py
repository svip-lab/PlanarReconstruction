import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceParameterLoss(nn.Module):
    def __init__(self, k_inv_dot_xy1):
        super(InstanceParameterLoss, self).__init__()
        self.k_inv_dot_xy1 = k_inv_dot_xy1

    def forward(self, segmentation, sample_segmentation, sample_params, valid_region, gt_depth, return_loss=True):
        """
        calculate loss of parameters
        first we combine sample segmentation with sample params to get K plane parameters
        then we used this parameter to infer plane based Q loss as done in PlaneRecover
        the loss enforce parameter is consistent with ground truth depth

        :param segmentation: tensor with size (h*w, K)
        :param sample_segmentation: tensor with size (N, K)
        :param sample_params: tensor with size (3, N), defined as n / d
        :param valid_region: tensor with size (1, 1, h, w), indicate planar region
        :param gt_depth: tensor with size (1, 1, h, w)
        :param return_loss: bool
        :return: loss
                 inferred depth with size (1, 1, h, w) corresponded to instance parameters
        """

        n = sample_segmentation.size(0)
        _, _, h, w = gt_depth.size()
        assert (segmentation.size(1) == sample_segmentation.size(1) and segmentation.size(0) == h*w
                and sample_params.size(1) == sample_segmentation.size(0))

        # combine sample segmentation and sample params to get instance parameters
        if not return_loss:
            sample_segmentation[sample_segmentation < 0.5] = 0.
        weight_matrix = F.normalize(sample_segmentation, p=1, dim=0)
        instance_param = torch.matmul(sample_params, weight_matrix)      # (3, K)

        # infer depth for every pixels and select the one with highest probability
        depth_maps = 1. / torch.matmul(instance_param.t(), self.k_inv_dot_xy1)     # (K, h*w)
        _, index = segmentation.max(dim=1)
        inferred_depth = depth_maps.t()[range(h*w), index].view(1, 1, h, w)

        if not return_loss:
            return _, inferred_depth, _, instance_param

        # select valid region
        valid_region = ((valid_region + (gt_depth != 0.0) ) == 2).view(-1)
        ray = self.k_inv_dot_xy1[:,  valid_region]                       # (3, N)
        segmentation = segmentation[valid_region]                        # (N, K)
        valid_depth = gt_depth.view(1, -1)[:, valid_region]              # (1, N)
        valid_inferred_depth = inferred_depth.view(1, -1)[:, valid_region]

        # Q_loss for every instance
        Q = valid_depth * ray                                          # (3, N)
        Q_loss = torch.abs(torch.matmul(instance_param.t(), Q) - 1.)   # (K, N)

        # weight Q_loss with probability
        weighted_Q_loss = Q_loss * segmentation.t()                    # (K, N)

        loss = torch.sum(torch.mean(weighted_Q_loss, dim=1))

        # abs distance for valid infered depth
        abs_distance = torch.mean(torch.abs(valid_inferred_depth - valid_depth))

        return loss, inferred_depth, abs_distance, instance_param
