import torch
import torch.nn as nn
from loguru import logger

from .backbone import *
from .head import hmr_head, smpl_head, smplcam_head, poco_head, pare_head, cliff_head
from .backbone.utils import get_backbone_info
from ..utils.train_utils import add_init_smpl_params_to_dict, load_statedict, get_part_statedict, get_model_path
from .head.poco_head import get_uncert_layer_info
from .head.nf_head import flow_head

class POCO(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            img_res=224,
            uncert_layer='diff_branch',
            activation_type='sigmoid',
            uncert_type=['pose'],
            uncert_inp_type='feat',
            loss_ver='gauss_sigma',
            num_neurons='1024-512',
            num_flow_layers=3,
            sigma_dim=9,
            num_nf_rv=9,
            mask_params_id='',
            nflow_mask_type='',
            exclude_uncert_idx='',
            use_dropout=False,
            use_iter_feats=False,
            cond_nflow=False,
            context_dim=1024,
            gt_pose_cond=False,
            gt_pose_cond_ds='h36m',
            gt_pose_cond_ratio=0.25,
            pretrained=None,
            inf_model='best',
            is_test=True,
    ):
        super(POCO, self).__init__()
        self.backbone_name, self.head_name = backbone.split('-')
        self.num_output_channels = get_backbone_info(self.backbone_name)['n_output_channels']
        self.uncert_layer = uncert_layer
        self.num_neurons = list(map(int,filter(None, num_neurons.split('-'))))
        self.num_flow_layers = num_flow_layers
        self.sigma_dim = sigma_dim
        self.num_nf_rv = num_nf_rv
        self.mask_params_id = mask_params_id
        self.nflow_mask_type = nflow_mask_type
        self.exclude_uncert_idx = list(filter(None, exclude_uncert_idx.split('-')))
        self.activation_type = activation_type
        self.use_dropout = use_dropout
        self.use_iter_feats = False if self.backbone_name.startswith('hrnet') else use_iter_feats
        self.uncert_type = uncert_type
        self.uncert_inp_type = uncert_inp_type
        self.cond_nflow = cond_nflow
        self.context_dim = context_dim
        self.gt_pose_cond = gt_pose_cond
        self.gt_pose_cond_ds = gt_pose_cond_ds
        self.gt_pose_cond_ratio = gt_pose_cond_ratio
        self.loss_ver = loss_ver
        self.inf_model = inf_model
        self.is_test = is_test

        self.backbone = eval(self.backbone_name)(pretrained=True)
        if self.head_name:
            self.head = eval(f'{self.head_name}_head(self.num_output_channels, self.uncert_layer, self.activation_type)')
        else:
            self.head = hmr_head(self.num_output_channels)

        if 'cliff' in self.head_name:
            self.smpl = smplcam_head(img_res=img_res)
        else:
            self.smpl = smpl_head(img_res=img_res)

        if 'diff_branch' in self.uncert_layer:
            self.uncert_head = poco_head(self.head.get_output_channels(),
                                         self.num_neurons,
                                         self.sigma_dim,
                                         self.activation_type,
                                         self.use_dropout,
                                         self.uncert_layer,
                                         self.exclude_uncert_idx,
                                         self.loss_ver,
                                         self.uncert_type,
                                         self.uncert_inp_type,
                                         self.gt_pose_cond,
                                         self.gt_pose_cond_ds,
                                         self.gt_pose_cond_ratio)

        if 'norm_flow' in self.loss_ver:
            self.flow_head = flow_head(self.uncert_type, self.num_flow_layers, \
                                       self.mask_params_id, self.nflow_mask_type, \
                                       self.exclude_uncert_idx, self.num_nf_rv, self.cond_nflow, \
                                       self.head.get_output_channels(), self.context_dim)
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

        if 'diff_branch' in self.uncert_layer:
            uncert_out = self.uncert_head(head_output, smpl_output, batch)
            smpl_output.update(uncert_out)

        if 'norm_flow' in self.loss_ver:
            flow_output = self.flow_head(head_output, smpl_output, batch)
            smpl_output.update(flow_output)

        return smpl_output

    def load_pretrained(self, file):
        pt_model_path = get_model_path(file, self.inf_model)
        logger.warning(f'Loading pretrained weights from {self.inf_model} {pt_model_path}')
        state_dict = torch.load(pt_model_path)
        state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict.keys() else state_dict

        if not ('init_pose' or 'init_shape' or 'init_cam') in state_dict.keys():
            state_dict = add_init_smpl_params_to_dict(state_dict)

        # Load Backbone
        load_statedict(self.backbone, state_dict, 'backbone')
        # Load Pose Head
        load_statedict(self.head, state_dict, 'head')
        # Load Uncertainty Head (if any)
        uncert_statedict = get_part_statedict(state_dict, 'uncert_head')
        if len(uncert_statedict.keys()) > 0 and self.uncert_layer == 'diff_branch' and self.is_test:
            logger.warning(f'Loading statedict of uncert_head for inference!!!')
            load_statedict(self.uncert_head, state_dict, 'uncert_head')
        # Load Normalizing flow head (if any)
        uncert_statedict = get_part_statedict(state_dict, 'flow_head')
        if len(uncert_statedict.keys()) > 0 and self.is_test:
            logger.warning(f'Loading statedict of flow_head for inference!!!')
            load_statedict(self.flow_head, state_dict, 'flow_head')
