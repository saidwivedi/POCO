import torch
import numpy as np
import torch.nn as nn
from loguru import logger

from ..layers import LocallyConnected2d
from ...utils.geometry import batch_rodrigues

def get_num_input_channels(num_input_channels, uncert_inp_type):
    if uncert_inp_type == 'feat-pose':
        num_input_channels += 24 * 3 * 3
    return num_input_channels

class poco_head(nn.Module):
    def __init__(
            self,
            num_input_channels,
            num_neurons,
            sigma_dim,
            activation_type,
            use_dropout,
            uncert_layer,
            exclude_uncert_idx,
            loss_ver,
            uncert_type,
            uncert_inp_type,
            gt_pose_cond,
            gt_pose_cond_ds,
            gt_pose_cond_ratio,
    ):
        super(poco_head, self).__init__()

        self.sigma_dim = sigma_dim if 'norm_flow' in loss_ver else 1
        self.activation_type = activation_type
        self.use_dropout = use_dropout
        self.uncert_layer = uncert_layer
        self.exclude_uncert_idx = exclude_uncert_idx
        self.loss_ver = loss_ver
        self.uncert_type = uncert_type
        self.uncert_inp_type = uncert_inp_type
        self.un_out = self.get_num_uncertainty_outputs()
        self.gt_pose_cond = gt_pose_cond
        self.gt_pose_cond_ds = gt_pose_cond_ds
        self.gt_pose_cond_ratio = gt_pose_cond_ratio

        if 'lc2d' in self.uncert_layer:
            self.uncert_lc2d = LocallyConnected2d(in_channels=num_input_channels,
                                                  out_channels=1,
                                                  output_size=[24, self.sigma_dim],
                                                  kernel_size=1,
                                                  stride=1)
            if self.activation_type == 'sigmoid':
                self.uncert_sigmoid = torch.nn.Sigmoid()
            elif self.activation_type == 'softplus':
                self.uncert_softplus = torch.nn.Softplus()
        else:
            num_input_channels = get_num_input_channels(num_input_channels,
                                                        self.uncert_inp_type)
            num_neurons.insert(0, num_input_channels)
            num_neurons.append(sum(self.un_out))
            self.num_neurons = num_neurons.copy()

            if 'pose-net' in self.uncert_inp_type:
                self.uncert_fc_poseNet = torch.nn.Linear(24*3*3, num_neurons[1])
                self.uncert_fc_poseNet_do = torch.nn.Dropout()
                self.uncert_fc_poseNet_act = torch.nn.Sigmoid()

                self.uncert_fc_featNet = torch.nn.Linear(num_neurons[0], num_neurons[1])
                self.uncert_fc_featNet_do = torch.nn.Dropout()
                self.uncert_fc_featNet_act = torch.nn.Sigmoid()
                num_neurons.pop(0)
                num_neurons[0] *= 2

            for layer_id in range(len(num_neurons)-1):
                inp_feat, out_feat = num_neurons[layer_id], num_neurons[layer_id+1]
                exec(f'self.uncert_fc{layer_id+1} = torch.nn.Linear({inp_feat}, {out_feat})')
                if self.use_dropout:
                    exec(f'self.uncert_dropout{layer_id+1} = torch.nn.Dropout()')
                if self.activation_type == 'sigmoid':
                    exec(f'self.uncert_sigmoid{layer_id+1} = torch.nn.Sigmoid()')
                elif self.activation_type == 'softplus':
                    exec(f'self.uncert_softplus{layer_id+1} = torch.nn.Softplus()')

    def get_num_uncertainty_outputs(self):
        num_outs = [0, 0, 0]
        self.num_uncert_parts = 24 - len(self.exclude_uncert_idx)
        if 'pose' in self.uncert_type:
            if self.loss_ver in ['genG', 'delta', 'mse_genG']:
                num_outs[0] = self.num_uncert_parts * 2 * self.sigma_dim
            elif self.loss_ver in 'gauss_genG':
                num_outs[0] = self.num_uncert_parts * 3 * self.sigma_dim
            else:
                num_outs[0] = self.num_uncert_parts * self.sigma_dim
        return num_outs

    def forward(self, head_output, smpl_output, batch):

        uncert_feats = head_output['uncert_feat']
        batch_size = uncert_feats.shape[0]
        num_neurons = self.num_neurons.copy()
        gt_pose_cond_idx = []
        if self.gt_pose_cond and 'is_train' in batch.keys():
            dt_names = np.array(batch['dataset_name'])
            gt_pose_cond_idx = np.ones(batch_size, dtype=bool)
            if self.gt_pose_cond_ds != 'all':
                gt_pose_cond_idx = np.where(dt_names == self.gt_pose_cond_ds)[0]
                gt_pose_cond_idx = gt_pose_cond_idx[:int(self.gt_pose_cond_ratio*len(gt_pose_cond_idx))]

        if 'lc2d' in self.uncert_layer:
            uncert_feats = self.uncert_lc2d(uncert_feats)
            if self.activation_type == 'sigmoid':
                uncert_feats = self.uncert_sigmoid(uncert_feats)
            elif self.activation_type == 'softplus':
                uncert_feats = self.uncert_softplus(uncert_feats)
        else:
            if 'pose' in self.uncert_inp_type:
                pose_inp = head_output['pred_pose'].reshape(batch_size, -1).clone()
                if self.gt_pose_cond and 'is_train' in batch.keys():
                    gt_pose = batch['pose'][gt_pose_cond_idx]
                    gt_pose = batch_rodrigues(gt_pose.view(-1,3)).view(-1,216)
                    pose_inp[gt_pose_cond_idx] = gt_pose
                if 'pose-net' in self.uncert_inp_type:
                    pose_feats = self.uncert_fc_poseNet(pose_inp)
                    pose_feats = self.uncert_fc_poseNet_do(pose_feats)
                    pose_feats = self.uncert_fc_poseNet_act(pose_feats)
                    uncert_feats = self.uncert_fc_featNet(uncert_feats)
                    uncert_feats = self.uncert_fc_featNet_do(uncert_feats)
                    uncert_feats = self.uncert_fc_featNet_act(uncert_feats)
                    uncert_feats = torch.cat([uncert_feats, pose_feats], 1)
                    num_neurons.pop(0)
                elif self.uncert_inp_type == 'feat-pose':
                    uncert_feats = torch.cat([uncert_feats, pose_inp], 1)

            for layer_id in range(len(num_neurons)-1):
                uncert_feats = eval(f'self.uncert_fc{layer_id+1}')(uncert_feats)
                if self.use_dropout:
                    uncert_feats = eval(f'self.uncert_dropout{layer_id+1}')(uncert_feats)
                if self.activation_type == 'sigmoid':
                    uncert_feats = eval(f'self.uncert_sigmoid{layer_id+1}')(uncert_feats)
                elif self.activation_type == 'softplus':
                    uncert_feats = eval(f'self.uncert_softplus{layer_id+1}')(uncert_feats)

        # Uncert Ouput
        output = {}
        st_idx, en_idx = 0, self.un_out[0]
        var_pose = uncert_feats[:,st_idx:en_idx]
        output['var_pose'] = var_pose.view(batch_size, -1, 3, 3) \
                                if self.sigma_dim == 9 else var_pose
        st_idx, en_idx = self.un_out[0], sum(self.un_out[:2])
        st_idx, en_idx = sum(self.un_out[:2]), sum(self.un_out)

        output['gt_pose_cond_idx'] = gt_pose_cond_idx

        return output

def get_uncert_layer_info(uncert_statedict):
    num_neurons = []
    for layer_name in uncert_statedict.keys():
        if 'weight' in layer_name:
            num_neurons.append(str(uncert_statedict[layer_name].shape[0]))
    num_neurons = num_neurons[:-1] # Skipping the last layer
    return num_neurons
