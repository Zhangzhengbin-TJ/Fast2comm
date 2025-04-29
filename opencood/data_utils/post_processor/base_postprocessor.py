# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Template for AnchorGenerator
"""

import numpy as np
import torch

from opencood.utils import box_utils


class BasePostprocessor(object):
    """
    Template for Anchor generator.

    Parameters
    ----------
    anchor_params : dict
        The dictionary containing all anchor-related parameters.
    train : bool
        Indicate train or test mode.

    Attributes
    ----------
    bbx_dict : dictionary
        Contain all objects information across the cav, key: id, value: bbx
        coordinates (1, 7)
    """

    def __init__(self, anchor_params, train=True):
        self.params = anchor_params
        self.bbx_dict = {}
        self.train = train

    def generate_anchor_box(self):
        # needs to be overloaded
        return None

    def generate_label(self, *argv):
        return None

    def generate_gt_bbx(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        # used to avoid repetitive bounding box
        object_id_list = []

        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space
            transformation_matrix = cav_content['transformation_matrix']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)

            # append the corresponding ids
            object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        # some of the bbx may be repetitive, use the id list to filter
        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

        return gt_box3d_tensor

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        from opencood.data_utils.datasets import GT_RANGE

        tmp_object_dict = {}
        for cav_content in cav_contents:  #遍历cav_contents，得到vehicles相关信息，放入tmp_object_dict中
            tmp_object_dict.update(cav_content['params']['vehicles'])

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else GT_RANGE  #训练时取['anchor_args']['cav_lidar_range']信息
        #Project the objects under world coordinates into another coordinat based on the provided extrinsic.
        #根据提供的外部坐标，将全局坐标下vehicle_bbox坐标投影到以ego为参考的坐标，
        #将投影后得到vehicle 3D bbox信息并存入output_dict中
        box_utils.project_world_objects(tmp_object_dict,  #vehicles相关信息
                                        output_dict,  #将得到的bbx_lidar信息放入output_dict中
                                        reference_lidar_pose, #The ego vehicle lidar pose under world coordinate.
                                        filter_range,  #cav_lidar_range信息
                                        self.params['order'])

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):  #遍历yaml里面的每一辆车的bbx
            object_np[i] = object_bbx[0, :]  #依次将object_np第i行赋值为为bbx参数
            mask[i] = 1  #依次将mask赋值为1，有多少个vehicle就有多少1
            object_ids.append(object_id)

        return object_np, mask, object_ids
