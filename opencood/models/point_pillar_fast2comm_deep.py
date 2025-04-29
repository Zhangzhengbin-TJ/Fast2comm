import torch.nn as nn
import torch

from logreplay.map.map_utils import convert_tl_status
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone,BaseBEVBackbone_deep_supervision
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.fuse_modules.fast2comm_fuse import Fast2comm
from opencood.models.fuse_modules.fast2comm_fuse_deep_supervision import Fast2commDeep
from opencood.models.fuse_modules.fast2comm_fuse_multihead import Fast2commMultiHead
from opencood.models.sub_modules.downsample_conv import DownsampleConv,DoubleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
import numpy as np
inference_results = []
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel,(8448,l1,c)
        x = self.att(x, x, x)
        x_fuse = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before,只取第一辆车(ego)的特征
        psm_fuse = x.permute(1, 2, 0).view(cav_num, C, H, W)  #融合后的所有特征
        return x_fuse, psm_fuse
        # return x_fuse
class PointPillarFast2commDeep(nn.Module):
    def __init__(self, args):
        super(PointPillarFast2commDeep, self).__init__()
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.deep_supervision_up = BaseBEVBackbone_deep_supervision(args['base_bev_backbone'], 2)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False
        ###用于深度监督
        self.multi_scale = args['Fast2comm_fusion']['multi_scale']
        if self.multi_scale:
            layer_nums = args['Fast2comm_fusion']['layer_nums']
            num_filters = args['Fast2comm_fusion']['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        ###
        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.fusion_net = Fast2commDeep(args['Fast2comm_fusion'])

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)
        self.cls_head_psm_fuse = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.cls_head_fuse_psm_single = nn.Conv2d(args['head_dim'] * self.num_levels, args['anchor_number'], kernel_size=1)
        self.channel_shrink_psm_fuse = nn.Conv2d(args['anchor_number'] * self.num_levels, args['anchor_number'],
                                                  kernel_size=1)
        self.channel_shrink_fuse_psm_single = nn.Conv2d(args['anchor_number'] * self.num_levels, args['anchor_number'],
                                        kernel_size=1)
        self.fused_feature_proj = DoubleConv(in_channels=384, out_channels=256, kernel_size=2, stride=2, padding=0)
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def feature_deep_supervision(self, x, B, record_len, deep_supervision_up=None):
        #不带上采样
        psm_single = self.cls_head(x)  # 生成预测结果，只有一个类别 (N,2,48,176)
        batch_psm_single = self.regroup(psm_single, record_len)  # (l1, 2, 48, 176)
        ups = []
        for i in range(self.num_levels):
            psm_single_list = []
            for b in range(B):
                neighbor_feature = batch_psm_single[b]
                psm_single_list.append(self.fuse_modules[i](neighbor_feature))
            psm_fuse = []
            fuse_psm_single = []
            for b in range(B):
                psm_fuse.append(psm_single_list[b][0])
                fuse_psm_single.append(psm_single_list[b][1])
            psm_fuse = torch.stack(psm_fuse)
            fuse_psm_single = torch.cat(fuse_psm_single, dim=0)

        if len(deep_supervision_up.deblocks_deep_supervision) > self.num_levels:
            psm_fuse = deep_supervision_up.deblocks_deep_supervision[-1](psm_fuse)
        return psm_fuse, fuse_psm_single

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,  #Voxel_feature：维度值分别表示voxel个数，voxel中的最大点云数量，
                                                            # 点云数据维度（x,y,z,r）
                      'voxel_coords': voxel_coords,  #Voxel_coords：维度值分别表示voxel个数，坐标值
                      'voxel_num_points': voxel_num_points,  #voxel个数
                      'record_len': record_len}
        label_dict = data_dict['label_dict']['gt_standup_2d_transfor2featuremap']  #取targets_label, (x1,y1,x2,y2,w,h)
        # n, 4 -> n, c, n是voxel的数量
        batch_dict = self.pillar_vfe(batch_dict) #得到pillar_features
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict) #得到spatial_features
        batch_dict = self.backbone(batch_dict) #得到spatial_features_2d

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        B = pairwise_t_matrix.shape[0]
        psm_fuse,fuse_psm_single = self.feature_deep_supervision(spatial_features_2d, B, record_len, self.deep_supervision_up)
        rm_sigle = self.reg_head(spatial_features_2d)
        # psm_single = self.cls_head(spatial_features_2d)  #生成预测结果，只有一个类别 (N,2,48,176)
        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            #fused_feature(4, 256, 48, 176)
            fused_feature, communication_rates = self.fusion_net(batch_dict['spatial_features'],
                                                                 fuse_psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                                 self.backbone)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
                                                                 fuse_psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix)

        psm = self.cls_head(fused_feature)  #（4,2,48,176）
        rm = self.reg_head(fused_feature)  #（4,14,48,176）
        # just for inference
        inference_results.append(communication_rates)
        if len(inference_results) == 2834:
            communication_rates = sum(inference_results) / len(inference_results)
            print(f"avg_communication_rates=:{communication_rates}")
        output_dict = {'psm_fuse': psm_fuse, 'psm': psm, 'rm': rm, 'com': communication_rates}#fast2comm
        # output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates}#ablation
        return output_dict
