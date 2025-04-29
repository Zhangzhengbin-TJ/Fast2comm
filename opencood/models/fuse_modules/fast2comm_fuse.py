"""
Implementation of Where2comm fusion.
"""
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
            # x = x * communication_masks_conf + x * communication_masks_gt  #相加
            x = torch.cat((x*communication_masks_conf, x*communication_masks_gt), dim=1) #拼接
            return x, communication_rates
        elif self.threshold:
            communication_rates = sum(communication_rates) / B
            # print(communication_rates)
            communication_masks = torch.cat(communication_masks, dim=0)
            if x.shape[-1] != communication_masks.shape[-1]:
                communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                    mode='bilinear', align_corners=False)  # (4,14,96,352)
            x = x * communication_masks
            x = torch.cat((x,x), dim=1)
            return x, communication_rates


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


class Fast2comm(nn.Module):
    def __init__(self, args):
        super(Fast2comm, self).__init__()
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
        self.DConv = nn.Sequential()
        self.DConv.append(nn.Sequential(nn.ConvTranspose2d(128,128, 1,1,bias=False),
                                        nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                                        nn.ReLU()
                                        ))
        self.DConv.append(nn.Sequential(nn.ConvTranspose2d(128,128, 2,2,bias=False),
                                        nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                                        nn.ReLU()
                                        ))
        self.DConv.append(nn.Sequential(nn.ConvTranspose2d(256,128, 4,4,bias=False),
                                        nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                                        nn.ReLU()
                                        ))

        c_in_list = [64, 128, 128]
        layer_strides = [2, 2, 2]
        self.Conv = nn.ModuleList()
        for idx in range(3):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.Conv.append(nn.Sequential(*cur_layers))


    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    def regroup_label(self, x, B):
        len = np.ones(B,dtype=np.int64)
        cum_sum_len = np.cumsum(len, axis=0)
        split_x = np.split(x, cum_sum_len[:-1])
        return split_x
    def visualize_feature(self, x, gt, record_len, B):
        x = self.regroup(x, record_len)
        for i in range(B):
            tmp, _  = torch.max(x[i], dim=1)
            tmp = tmp.cpu()
            if tmp.size()[0]==1:
                tmp = tmp[0].reshape(96, 352)
            else:
                tmp = tmp[1].reshape(96,352)
            plt.imshow(tmp.detach().numpy(), cmap='hot',interpolation='nearest')
            # plt.show()
            # # plt.imshow(max_vaule[i][1].detach().numpy(), cmap='hot', interpolation='nearest')
            # # plt.show()
            for box in gt[i]:
                x1, y1, x2, y2 = box[:4]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=1, edgecolor='blue', facecolor='none')
                plt.gca().add_patch(rect)
            # plt.colorbar()  # 添加颜色条
            # plt.title('Tensor Heatmap with Boxes')
            plt.axis('off')
            plt.savefig('/media/zzbhhh/039c7a76-c72c-4261-af64-a755a33e3610/zzb/Code/fast2comm_vis/img.png',
                        dpi=600,
                        bbox_inches='tight',
                        pad_inches=0)
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
                x = self.Conv[i](x)

                # 1. Communication (mask the features)
                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                    else:
                        # Prune
                        batch_confidence_maps = self.regroup(fuse_psm_single, record_len)  #(l1,2,48,176),...
                        batch_rm_sigle = self.regroup(rm_sigle, record_len)  #(l1,14,48,176),...
                        # batch_targets_label = self.regroup_label(targets_label, B)  ##(1,14,48,176),...
                        #由于使用record_len对只对一个,产生mask（只有0和1），communication_rates就是将为1的地方相加除以整个特征大小'

                        x, communication_rates = self.naive_communication(x,
                                                                          batch_confidence_maps,
                                                                          batch_rm_sigle,
                                                                          targets_label,
                                                                          B)

                # 2. Split the features
                # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
                # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
                batch_node_features = self.regroup(x, record_len)

                # 3. Fusion
                x_fuse = []
                for b in range(B):
                    neighbor_feature = batch_node_features[b]
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                # 4. Deconv
                if len(self.DConv) > 0:
                    ups.append(self.DConv[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
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
