# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import cv2
from loguru import logger
import numpy as np
import torch
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from ..utils.smooth_bbox import get_all_bbox_params
from ..utils.image_utils import calculate_bbox_info, calculate_focal_length
from ..utils.vibe_image_utils import get_single_image_crop_demo, points_inside_bbox


class Inference(Dataset):
    def __init__(self, image_folder, frames=None, bboxes=None, joints2d=None, frame_kps=None,
                 scale=1.0, crop_size=224, return_dict=False, normalize_kp2d=False):
        self.image_file_names = sorted([
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        ])
        self.image_file_names = sorted(self.image_file_names)

        self.image_file_names = np.array(self.image_file_names) \
            if frames is None else np.array(self.image_file_names)[frames]

        self.scale = scale
        self.bboxes = bboxes
        self.frames = frames
        self.joints2d = joints2d
        self.crop_size = crop_size
        self.return_dict = return_dict
        self.normalize_kp2d = normalize_kp2d
        self.has_keypoints = True if joints2d is not None else False

        self.has_frame_kps = True if frame_kps is not None else False
        self.frame_kps = frame_kps

        self.norm_joints2d = np.zeros_like(self.joints2d)

        if self.has_keypoints:
            bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
            bboxes[:, 2:] = 150. / bboxes[:, 2:]
            self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

            self.image_file_names = self.image_file_names[time_pt1:time_pt2]
            self.joints2d = joints2d[time_pt1:time_pt2]

            if frames is not None:
                self.frames = frames[time_pt1:time_pt2]

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)

        orig_shape = np.array(img.shape)[:2]

        bbox = self.bboxes[idx]
        center = bbox[:2]
        scale = max(bbox[2], bbox[3])/200

        focal_length = calculate_focal_length(orig_shape[0], orig_shape[1])
        bbox_info = calculate_bbox_info(center, scale, orig_shape)

        j2d = self.joints2d[idx] if self.has_keypoints else None

        try:
            frame_kps = self.frame_kps[self.frames[idx]]
            for person_id in frame_kps.keys():
                person_kps = frame_kps[person_id]
                if points_inside_bbox(person_kps[:,:2], bbox):
                    op_keypoints = person_kps
                    break
        except:
            op_keypoints = None
            # logger.warning(f'Openpose keypoints not found for frame--{idx}')

        # If bbox detection and optimization is enabled, use openpose jnts as j2d
        if self.has_frame_kps:
            if isinstance(op_keypoints, np.ndarray):
                extra_keypoints2d = np.zeros((19, 3))
                j2d = np.concatenate((op_keypoints, extra_keypoints2d), axis=0).astype(np.float32)
            else:
                j2d = np.zeros((44, 3)).astype(np.float32)
            kp_2d = j2d.copy()
        else:
            kp_2d = None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=kp_2d,
            scale=self.scale,
            crop_size=self.crop_size
        )

        if self.normalize_kp2d:
            kp_2d[:, :-1] = 2. * kp_2d[:, :-1] / self.crop_size - 1.

        if kp_2d is None:
            kp_2d = torch.empty(1)
        if j2d is None:
            j2d = torch.empty(1)

        return {
            'has_frame_kps': self.has_frame_kps,
            'has_keypoints': self.has_keypoints,
            'img' : norm_img,
            'j2d': kp_2d,
            'j2d_orig': j2d,
            'bbox_info': bbox_info,
            'focal_length': focal_length,
            'scale': scale,
            'center': center,
            'orig_shape': orig_shape,
        }


class ImageFolder(Dataset):
    def __init__(self, image_folder, bboxes=None, joints2d=None,
                 scale=1.0, crop_size=224, return_dict=False, normalize_kp2d=False):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)

        self.scale = scale
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.crop_size = crop_size
        self.return_dict = return_dict
        self.normalize_kp2d = normalize_kp2d
        self.has_keypoints = True if joints2d is not None else False

        self.norm_joints2d = np.zeros_like(self.joints2d)

        if self.has_keypoints:
            bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
            bboxes[:, 2:] = 150. / bboxes[:, 2:]
            self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

            self.image_file_names = self.image_file_names[time_pt1:time_pt2]
            self.joints2d = joints2d[time_pt1:time_pt2]

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)

        bbox = self.bboxes[idx]

        j2d = self.joints2d[idx] if self.has_keypoints else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=self.scale,
            crop_size=self.crop_size
        )

        if self.normalize_kp2d:
            kp_2d[:, :-1] = 2. * kp_2d[:, :-1] / self.crop_size - 1.

        if self.has_keypoints:
            if self.return_dict:
                return {
                    'disp_img': raw_img,
                    'img': norm_img,
                    'keypoints': kp_2d,
                }
            else:
                return norm_img, kp_2d
        else:
            return norm_img
