"""
Implementation of Where2comm fusion.
"""
import argparse
from unittest.mock import patch

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from numba.tests.test_gil import sleep
from torch.cuda import device_of
from torch.nn.functional import selu_
from torch.onnx.symbolic_opset9 import tensor

from docs.conf import project
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
from opencood.models.fuse_modules.multi_head_selfattn import MuiltHeadAtten
from opencood.models.sub_modules.downsample_conv import DownsampleConv
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
        self.transfor = nn.Conv2d(128,64,kernel_size=1, stride=1, padding=0)

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()
    def create_mask(self, mask, gt_label_feature):
        '''
        Creating communication mask from mask straightly.
        args:
            mask:all zero:(l1, h, w)
            gt_label_feature.list:(num_cav, 6)
        return: torch.tensor: updated mask
        '''
        for feature in gt_label_feature:
            x1, y1, x2, y2 = feature[:4]
            x1, x2 = max(0, x1), min(mask.shape[2], x2)
            y1, y2 = max(0, y1), min(mask.shape[1], y2)
            mask[:, y1:y2, x1:x2] = 1
        return mask

    def forward(self, x, batch_confidence_maps, batch_rm_sigle, batch_targets_label, B):
        #batch_confidence_maps(l1,2,48,176),(l2,2,48,176)(l3,2,48,176), (l4,2,48,176)
        #batch_targets_label:(l1,14,48,176),...
        #batch_rm_sigle:(l1,14,48,176)
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """

        _, _, H, W = batch_confidence_maps[0].shape

        communication_masks_conf = []
        communication_masks_gt = []
        communication_masks = []
        communication_rates = []
        # for b in range(B):
        #     ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)  #取维度1的最大数字
        #     if self.smooth:
        #         communication_maps = self.gaussian_filter(ori_communication_maps)
        #     else:
        #         communication_maps = ori_communication_maps
        #     # communication_maps = ori_communication_maps
        #
        #     L = communication_maps.shape[0]
        #     if self.training:
        #         # #Official training proxy objective
        #         K = int(H * W * random.uniform(0, 1))  #既然是选择前景，那为什么k是随机的？改变前景的选择方式
        #         communication_maps_conf = communication_maps.reshape(L, H * W)  #communication_maps值为L,H*W
        #         _, indices = torch.topk(communication_maps_conf, k=K, sorted=False)  #得到k个最大值的索引
        #         communication_mask_conf = torch.zeros_like(communication_maps_conf).to(communication_maps_conf.device)  #(l1,8448)
        #         ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps_conf.device)
        #         # communication_mask:(l1,1,48,176)
        #         communication_mask_conf = torch.scatter(communication_mask_conf, -1, indices, ones_fill).reshape(L, 1, H, W)
        #
        #         #fast2comm training
        #         communication_maps_gt = communication_maps.reshape(L, H, W)  # communication_maps值为L,H,W
        #         communication_mask_zero = torch.zeros_like(communication_maps_gt).to(communication_maps_gt.device)  # (l1,8448)
        #         communication_mask_gt = self.create_mask(communication_mask_zero, batch_targets_label[b])
        #         communication_mask_gt = communication_mask_gt.reshape(L, 1, H, W)
        #     elif self.threshold:
        #         ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
        #         zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
        #         communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
        #     else:
        #         communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)
        #     if self.training:
        #         communication_rate = communication_mask_conf.sum() / (L * H * W) + communication_mask_gt.sum() / (L * H * W)
        #         # Ego
        #         communication_mask_conf[0] = 1
        #         communication_mask_gt[0] = 1
        #
        #         communication_masks_conf.append(communication_mask_conf)
        #         communication_masks_gt.append(communication_mask_gt)
        #         communication_rates.append(communication_rate)
        #     elif self.threshold:
        #         communication_rate = communication_mask.sum() / (L * H * W)
        #         # Ego
        #         communication_mask[0] = 1
        #
        #         communication_masks.append(communication_mask)
        #         communication_rates.append(communication_rate)
        # if self.training:
        #     communication_rates = sum(communication_rates) / B
        #     # print(communication_rates)
        #     communication_masks_conf = torch.cat(communication_masks_conf, dim=0)
        #     communication_masks_gt = torch.cat(communication_masks_gt, dim=0)
        #     if x.shape[-1] != communication_masks_conf.shape[-1]:
        #         communication_masks_conf = F.interpolate(communication_masks_conf, size=(x.shape[-2], x.shape[-1]),
        #                                             mode='bilinear', align_corners=False)
        #         communication_masks_gt = F.interpolate(communication_masks_gt, size=(x.shape[-2], x.shape[-1]),
        #                                             mode='bilinear', align_corners=False)
        #     # x = x * communication_masks_conf + x * communication_masks_gt
        #     x = torch.cat((x*communication_masks_conf, x*communication_masks_gt), dim=1)
        #     x = self.transfor(x)
        #     return x, communication_rates
        # elif self.threshold:
        #     communication_rates = sum(communication_rates) / B
        #     print(communication_rates)
        #     communication_masks = torch.cat(communication_masks, dim=0)
        #     if x.shape[-1] != communication_masks.shape[-1]:
        #         communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
        #                                             mode='bilinear', align_corners=False)  # (4,14,96,352)
        #     x = x * communication_masks
        #     return x, communication_rates
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)  #取维度1的最大数字
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps
            # communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                # #Official training proxy objective
                K = int(H * W * random.uniform(0, 1))  #既然是选择前景，那为什么k是随机的？改变前景的选择方式
                communication_maps_conf = communication_maps.reshape(L, H * W)  #communication_maps值为L,H*W
                _, indices = torch.topk(communication_maps_conf, k=K, sorted=False)  #得到k个最大值的索引
                communication_mask_conf = torch.zeros_like(communication_maps_conf).to(communication_maps_conf.device)  #(l1,8448)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps_conf.device)
                # communication_mask:(l1,1,48,176)
                communication_mask_conf = torch.scatter(communication_mask_conf, -1, indices, ones_fill).reshape(L, 1, H, W)

                #fast2comm training
                communication_maps_gt = communication_maps.reshape(L, H, W)  # communication_maps值为L,H,W
                communication_mask_zero = torch.zeros_like(communication_maps_gt).to(communication_maps_gt.device)  # (l1,8448)
                communication_mask_gt = self.create_mask(communication_mask_zero, batch_targets_label[b])
                communication_mask_gt = communication_mask_gt.reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            if self.training:
                communication_rate = communication_mask_conf.sum() / (L * H * W) + communication_mask_gt.sum() / (L * H * W)
                # Ego
                communication_mask_conf[0] = 1
                communication_mask_gt[0] = 1

                communication_masks_conf.append(communication_mask_conf)
                communication_masks_gt.append(communication_mask_gt)
                communication_rates.append(communication_rate)
            elif self.threshold:
                communication_rate = communication_mask.sum() / (L * H * W)
                # Ego
                communication_mask[0] = 1

                communication_masks.append(communication_mask)
                communication_rates.append(communication_rate)
        if self.training:
            communication_rates = sum(communication_rates) / B
            # print(communication_rates)
            communication_masks_conf = torch.cat(communication_masks_conf, dim=0)
            communication_masks_gt = torch.cat(communication_masks_gt, dim=0)
            if x.shape[-1] != communication_masks_conf.shape[-1]:
                communication_masks_conf = F.interpolate(communication_masks_conf, size=(x.shape[-2], x.shape[-1]),
                                                    mode='bilinear', align_corners=False)
                communication_masks_gt = F.interpolate(communication_masks_gt, size=(x.shape[-2], x.shape[-1]),
                                                    mode='bilinear', align_corners=False)
            x_conf = x * communication_masks_conf
            x_gt = x * communication_masks_gt
            # x = torch.cat((x*communication_masks_conf, x*communication_masks_gt), dim=1)
            # x = self.transfor(x)
            return x_conf, x_gt, communication_rates
        elif self.threshold:
            communication_rates = sum(communication_rates) / B
            # print(communication_rates)
            communication_masks = torch.cat(communication_masks, dim=0)
            if x.shape[-1] != communication_masks.shape[-1]:
                communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                    mode='bilinear', align_corners=False)  # (4,14,96,352)
            x = x * communication_masks
            return x, x, communication_rates


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, q, k, v):
        cav_num, C, H, W = q.shape
        q = q.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        k = k.view(cav_num, C, -1).permute(2, 0, 1)
        v = v.view(cav_num, C, -1).permute(2, 0, 1)
        x = self.att(q, k, v)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class Fast2commMultiHead(nn.Module):
    def __init__(self, args):
        super(Fast2commMultiHead, self).__init__()
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
        # if self.multi_scale:
        #     layer_nums = args['layer_nums']
        #     num_filters = args['num_filters']
        #     self.num_levels = len(layer_nums)
        #     self.fuse_modules = nn.ModuleList()
        #     for idx in range(self.num_levels):
        #         fuse_network = MobileViTCood(in_channels=num_filters[idx],
        #                                  attn_unit_dim=num_filters[idx],
        #                                  )
        #         self.fuse_modules.append(fuse_network)
        # else:
        #     self.fuse_modules = AttentionFusion(args['in_channels'])

        '''这里是使用patch_embed方法'''
        # patch_size = 4
        # self.naive_communication = Communication(args['communication'])
        # self.proj = nn.Sequential()
        # self.proj.append(nn.Conv2d(64, out_channels=256, kernel_size=patch_size, stride=patch_size))
        # self.proj.append(nn.Conv2d(128, out_channels=256, kernel_size=patch_size, stride=patch_size))
        # self.proj.append(nn.Conv2d(256, out_channels=256, kernel_size=patch_size, stride=patch_size))
        #
        # self.deconv = nn.Sequential()
        # self.deconv.append(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=patch_size, stride=patch_size))
        # self.deconv.append(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=patch_size*2, stride=patch_size*2))
        # self.deconv.append(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=patch_size*4, stride=patch_size*4))
        #
        # self.pos = nn.ModuleList()
        # self.pos.append(nn.Embedding(num_embeddings=2112, embedding_dim=1024))
        # self.pos.append(nn.Embedding(num_embeddings=528, embedding_dim=2048))
        # self.pos.append(nn.Embedding(num_embeddings=132, embedding_dim=4096))
        # # self.multi_head_attn = MuiltHeadAtten()
        # self.multi_head_attn = nn.ModuleList()
        # for i in range(self.num_levels):
        #     self.multi_head_attn.append(MuiltHeadAtten(d_model=num_filters[i]*patch_size*patch_size))
        '''通道变换'''
        patch_size = 4
        self.naive_communication = Communication(args['communication'])
        self.proj = nn.Sequential()
        self.proj.append(nn.Conv2d(64, out_channels=64, kernel_size=patch_size, stride=patch_size))
        self.proj.append(nn.Conv2d(128, out_channels=128, kernel_size=patch_size, stride=patch_size))
        self.proj.append(nn.Conv2d(256, out_channels=256, kernel_size=patch_size, stride=patch_size))

        self.deconv = nn.Sequential()
        self.deconv.append(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=patch_size, stride=patch_size))
        self.deconv.append(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=patch_size * 2, stride=patch_size * 2))
        self.deconv.append(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=patch_size * 4, stride=patch_size * 4))

        self.pos = nn.ModuleList()
        self.pos.append(nn.Embedding(num_embeddings=2112, embedding_dim=64))
        self.pos.append(nn.Embedding(num_embeddings=528, embedding_dim=128))
        self.pos.append(nn.Embedding(num_embeddings=132, embedding_dim=256))
        # self.multi_head_attn = MuiltHeadAtten()
        self.multi_head_attn = nn.ModuleList()
        for i in range(self.num_levels):
            self.multi_head_attn.append(MuiltHeadAtten(d_model=num_filters[i]))

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    def regroup_label(self, x, B):
        len = np.ones(B,dtype=np.int64)
        cum_sum_len = np.cumsum(len, axis=0)
        split_x = np.split(x, cum_sum_len[:-1])
        return split_x
    def visualize_feature(self, x, gt, record_len, B, communication_masks):
        x = self.regroup(x, record_len)
        communication_masks = self.regroup(communication_masks, record_len)
        for i in range(B):
            # tmp, _ = torch.max(communication_masks[i], dim=0)
            tmp = communication_masks[i].cpu()
            tmp = tmp[1].view(96,352)
            plt.imshow(tmp.detach().numpy(), cmap='hot', interpolation='nearest')
            plt.show()
        for i in range(B):
            tmp, _  = torch.max(x[i], dim=1)
            tmp = tmp.cpu()
            tmp = tmp[1].view(96,352)
            plt.imshow(tmp.detach().numpy(), cmap='hot',interpolation='nearest')
            # plt.show()
            # # plt.imshow(max_vaule[i][1].detach().numpy(), cmap='hot', interpolation='nearest')
            # # plt.show()
            for box in gt[i]:
                x1, y1, x2, y2 = box[:4]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=1, edgecolor='blue', facecolor='none')
                plt.gca().add_patch(rect)
            plt.colorbar()  # 添加颜色条
            plt.title('Tensor Heatmap with Boxes')
            plt.show()

    def forward(self, x, fuse_psm_single, rm_sigle, targets_label, record_len, pairwise_t_matrix, backbone=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """

        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]
        if self.multi_scale:
            ups = []

            for i in range(self.num_levels):
                x = backbone.blocks[i](x)  #(n,64,96,352),(n,128,48,176),(n,256,24,88)
                # 1. Communication (mask the features)
                # if i == 0:
                if self.fully:
                    communication_rates = torch.tensor(1).to(x.device)
                else:
                    # Prune
                    batch_confidence_maps = self.regroup(fuse_psm_single, record_len)  #(l1,2,48,176),...
                    batch_rm_sigle = self.regroup(rm_sigle, record_len)  #(l1,14,48,176),...
                    # batch_targets_label = self.regroup_label(targets_label, B)  ##(1,14,48,176),...
                    #由于使用record_len对只对一个,产生mask（只有0和1），communication_rates就是将为1的地方相加除以整个特征大小'

                    x_conf, x_gt, communication_rates = self.naive_communication(x,
                                                                      batch_confidence_maps,
                                                                      batch_rm_sigle,
                                                                      targets_label,
                                                                      B)
                        # if x.shape[-1] != communication_masks.shape[-1]:
                        #     communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                        #                                         mode='bilinear', align_corners=False)  #(4,14,96,352)
                        #where2comm
                        # x = x*communication_masks
                        # self.visualize_feature(x, targets_label, record_len, B,
                        #                        communication_masks)  # 可视化特征图
                        #fast2comm
                        # x_match_mask = []
                        # batch_x = self.regroup(x,record_len)
                        # for b in range(B):
                        #     tmp, _ = batch_x[b].sigmoid().max(dim=0, keepdim=True)
                        #     x_match_mask.append(tmp)
                        # x_match_mask = torch.cat(x_match_mask, dim=0)
                        # x = x_match_mask * communication_masks
                #patch embedding
                x_ori = self.proj[i](x)
                x_conf = self.proj[i](x_conf)
                x_gt = self.proj[i](x_gt)

                # 2. Split the features
                # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
                # For example [[2, 64, 48, 176], [1, 64, 48, 176], ...]
                #where2comm
                batch_node_features_x = self.regroup(x_ori, record_len)
                batch_node_features_conf = self.regroup(x_conf, record_len) #置信图产生的特征
                batch_node_features_gt = self.regroup(x_gt, record_len) #根据标签产生的特征
                #fast2comm
                #batch_node_features = self.regroup_label(x, B)

                # 3. Fusion
                x_fuse = []
                for b in range(B):
                    neighbor_feature_x = batch_node_features_x[b]
                    neighbor_feature_conf = batch_node_features_conf[b]
                    neighbor_feature_gt = batch_node_features_gt[b]
                    pos = self.pos[i].weight
                    pos = pos.unsqueeze(1).repeat(1, neighbor_feature_conf.shape[0], 1)
                    if self.training:
                        x_fuse.append(self.multi_head_attn[i](neighbor_feature_x,
                                                              neighbor_feature_conf,
                                                              neighbor_feature_gt,
                                                              pos))
                    else:
                        x_fuse.append(self.multi_head_attn[i](neighbor_feature_conf,
                                                              neighbor_feature_conf,
                                                              neighbor_feature_gt,
                                                              pos))
                    # if self.training:
                    #     x_fuse.append(self.fuse_modules[i](neighbor_feature_conf, neighbor_feature_gt, neighbor_feature_x))
                    # else:
                    #     x_fuse.append(
                    #         self.fuse_modules[i](neighbor_feature_conf, neighbor_feature_gt, neighbor_feature_conf))
                x_fuse = torch.stack(x_fuse) #(bs,1024,24,88)
                # 4. Deconv
                if len(backbone.deblocks) > 0:
                    ups.append(self.deconv[i](x_fuse))
                    # ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
        else:
            # 1. Communication (mask the features)
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
            else:
                # Prune
                batch_confidence_maps = self.regroup(fuse_psm_single, record_len)
                batch_targets_label = self.regroup(targets_label, B)
                communication_masks, communication_rates = self.naive_communication(batch_confidence_maps,
                                                                                    batch_targets_label,
                                                                                    B)
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
