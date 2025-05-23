"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):#教程：https://blog.csdn.net/Jack_Man_N/article/details/134613118
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']
        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3 #num_point_features实际上是作为卷积的输入通道使用

        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):

        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']
        #  计算点云的平均位置，并将每个点的坐标减去平均位置得到聚类特征(f_cluster)
        points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean
        #  计算点云中心特征(f_center)，通过将每个点的x、y、z坐标减去相应的体素偏移量得到
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_z + self.z_offset)
        #  根据"use_absolute_xyz"标志位的设置，选择不同的特征组合方式。如果设置为True，将使用原始的点云特征、
        #  聚类特征和中心特征作为输入特征。否则，将只使用点云特征的后3个维度（排除坐标特征），并与聚类特征和中心特征一起作为输入特征。
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        #  如果设置了"with_distance"标志位，计算点云特征的欧几里得距离，并将其作为额外的特征添加到输入特征中。
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                     keepdim=True)
            features.append(points_dist)
        #  将所有的特征按最后一个维度拼接起来，形成最终的输入特征
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                           axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
