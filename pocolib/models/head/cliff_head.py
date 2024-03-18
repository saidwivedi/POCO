import torch
import numpy as np
import torch.nn as nn

from ...core.config import SMPL_MEAN_PARAMS
from ...utils.geometry import rot6d_to_rotmat

BN_MOMENTUM = 0.1

class cliff_head(nn.Module):
    def __init__(
            self,
            num_input_features,
            uncert_layer='',
            uncert_act='',
            num_joints=24,

    ):
        super(cliff_head, self).__init__()
        self.num_joints = num_joints
        npose = num_joints * 6
        self.npose = npose
        self.num_input_features = num_input_features

        num_input_features += 3   #bbox

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_input_features + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        # self.downsample_module = self._make_head()

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:num_joints*6]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_head(self):
        # downsampling modules
        downsamp_modules = []
        for i in range(3):
            in_channels = self.num_input_features
            out_channels = self.num_input_features

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)

        downsamp_modules = nn.Sequential(*downsamp_modules)

        return downsamp_modules

    def forward(
            self,
            features,
            batch,
            n_iter=3,
            init_pose=None,
            init_shape=None,
            init_cam=None,
    ):

        batch_size = features.shape[0]
        bbox_info = batch['bbox_info']

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        # Not needed if HRNET_cls model is used
        if features.dim() > 2:
            features = self.avgpool(features)
            features = features.view(features.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for i in range(n_iter):
            # print(features.shape, bbox_info.shape)
            xc = torch.cat([features, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size,
                                                      self.num_joints, 3, 3)

        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose_6d': pred_pose,
            'uncert_feat': features,
            'body_feat2': xc,
        }

        return output

    def get_output_channels(self):
        # body_feat -- Nx2048
        # body_feat2 -- Nx1024
        return 2048

