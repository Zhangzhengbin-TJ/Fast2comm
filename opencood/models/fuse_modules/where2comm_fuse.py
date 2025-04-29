"""
Implementation of Where2comm fusion.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.core.computation.expr import intersection

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()
    def calculate_feature_overlap(self, feature):  #iou计算，只有特征值为1时有效
            feature = feature.cpu()
            ones_mask = torch.ones_like(feature).cpu()
            zeros_mask = torch.zeros_like(feature).cpu()
            communication_mask = torch.where(feature > self.threshold, ones_mask, zeros_mask).cpu()
            ious = []
            for i in range(communication_mask.shape[0]):
                for j in range(i+1, communication_mask.shape[0]):
                    feature1 = communication_mask[i][0].view(-1).cpu()
                    feature2 = communication_mask[j][0].view(-1).cpu()
                    feature1_inds = feature1 == 1
                    feature2_inds = feature2 == 1
                    intersection = (feature2_inds[feature1_inds]).long().sum().data.cpu().numpy()
                    union = feature1_inds.long().sum().data.cpu().numpy()+feature2_inds.long().sum().data.cpu().numpy()-intersection
                    # union = np.logical_or(feature1==1, feature2==1).cpu()
                    iou = float(intersection)/float(max(union, 1))
                    if iou == 0:
                        ious.append(float('nan'))
                    else:
                        ious.append(iou)
                    print(iou)



    def forward(self, batch_confidence_maps, B):  #batch_confidence_maps(l1,2,48,176),(l2,2,48,176)(l3,2,48,176),
                                                    # (l4,2,48,176)
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """

        _, _, H, W = batch_confidence_maps[0].shape

        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)  #取维度1的最大数字
                                                                                                     #(l1,1,48,176)
            # ori_communication_maps, _ = batch_confidence_maps[b].softmax().max(dim=1, keepdim=True)  # 取维度1的最大数字
            # # (l1,1,48,176)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps
            # self.calculate_feature_overlap(communication_maps)

            L = communication_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0, 1))  #既然是选择前景，那为什么k是随机的？改变前景的选择方式
                communication_maps = communication_maps.reshape(L, H * W)  #communication_maps值为L,H*W
                _, indices = torch.topk(communication_maps, k=K, sorted=False)  #得到k个最大值的索引
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)  #(l1,8448)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
                #communication_mask:(l1,1,48,176)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            communication_rate = communication_mask.sum() / (L * H * W)
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        # print(communication_rates)
        return communication_masks, communication_rates


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')

        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttentionFusion(args['in_channels'])

        self.naive_communication = Communication(args['communication'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, psm_single, record_len, pairwise_t_matrix, backbone=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """

        _, C, H, W = x.shape #(n,64,192,704)
        B = pairwise_t_matrix.shape[0]

        if self.multi_scale:
            ups = []

            for i in range(self.num_levels):
                x = backbone.blocks[i](x) #(n,64,96,352)

                # 1. Communication (mask the features)
                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                    else:
                        # Prune
                        batch_confidence_maps = self.regroup(psm_single, record_len)  #(l1,2,48,176),...

                        #由于使用record_len对只对一个,产生mask（只有0和1），communication_rates就是将为1的地方相加除以整个特征大小
                        communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
                        if x.shape[-1] != communication_masks.shape[-1]:
                            communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                                mode='bilinear', align_corners=False)
                        x = x * communication_masks

                # 2. Split the features
                # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
                # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
                batch_node_features = self.regroup(x, record_len)

                # 3. Fusion
                x_fuse = []
                for b in range(B):
                    neighbor_feature = batch_node_features[b]
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)  #(bs, 64, 96,352)

                # 4. Deconv
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse)) #list{3}:(bs,128,96,352)
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1) #(bs,128x3,96,352)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            # 1. Communication (mask the features)
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
            else:
                # Prune
                batch_confidence_maps = self.regroup(psm_single, record_len)
                communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
                x = x * communication_masks

            # 2. Split the features
            # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
            # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
            batch_node_features = self.regroup(x, record_len)

            # 3. Fusion
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        return x_fuse, communication_rates
