import torch
import torch.nn as nn
from loguru import logger

from .backbone import *
from .head import hmr_head, smpl_head, smplcam_head, pare_head, cliff_head
from .backbone.utils import get_backbone_info
from ..utils.train_utils import add_init_smpl_params_to_dict, load_statedict, get_part_statedict, get_model_path

class HMR(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            img_res=224,
            pretrained=None,
            inf_model='best',
            is_test=True,
    ):
        super(HMR, self).__init__()
        self.backbone_name, self.head_name = backbone.split('-')
        self.num_output_channels = get_backbone_info(self.backbone_name)['n_output_channels']
        self.backbone = eval(self.backbone_name)(pretrained=True)
        self.inf_model = inf_model
        if self.head_name:
            self.head = eval(f'{self.head_name}_head(self.num_output_channels)')
        else:
            self.head = hmr_head(self.num_output_channels)
        if 'cliff' in self.head_name:
            self.smpl = smplcam_head(img_res=img_res)
        else:
            self.smpl = smpl_head(img_res=img_res)
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(self, batch):
        features = self.backbone(batch['img'])
        if 'cliff' in self.head_name:
            head_output = self.head(features, batch)
            smpl_output = self.smpl(
                rotmat=head_output['pred_pose'],
                shape=head_output['pred_shape'],
                cam=head_output['pred_cam'],
                focal_length=batch['focal_length'],
                bbox_scale=batch['scale'],
                bbox_center=batch['center'],
                img_h=batch['orig_shape'][:,0],
                img_w=batch['orig_shape'][:,1])
        else:
            head_output = self.head(features)
            smpl_output = self.smpl(
                rotmat=head_output['pred_pose'],
                shape=head_output['pred_shape'],
                cam=head_output['pred_cam'],
                normalize_joints2d=True)
        smpl_output.update(head_output)
        return smpl_output

    def load_pretrained(self, file):
        pt_model_path = get_model_path(file, self.inf_model)
        logger.warning(f'Loading pretrained weights from {file}')
        state_dict = torch.load(pt_model_path)
        state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict.keys() else state_dict

        if not ('init_pose' or 'init_shape' or 'init_cam') in state_dict.keys():
            logger.warning(f'Adding mean init pose/shape/cam')
            state_dict = add_init_smpl_params_to_dict(state_dict)

        load_statedict(self.backbone, state_dict, 'backbone')
        load_statedict(self.head, state_dict, 'head')
