import math
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as distributions
from loguru import logger

from ..layers.real_nvp import RealNVP
from ...utils.geometry import batch_rodrigues, matrix_to_rotation_6d


# NN for scaling and trans
def net_s(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(), nn.Linear(h, h), nn.LeakyReLU(), nn.Linear(h, o), nn.Tanh())

def net_t(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(), nn.Linear(h, h), nn.LeakyReLU(), nn.Linear(h, o))

# Masks for Nflow
def get_alter_masks(num_nf_rv, num_flow_layers):
    return np.array([list(i%2 for i in range(num_nf_rv)), list((i+1)%2 for i in reversed(range(num_nf_rv)))] * num_flow_layers).astype(np.float32)

def get_new_masks(num_nf_rv, num_flow_layers):
    rv_split = math.floor(num_nf_rv/2)
    return np.array([list(min(i//rv_split,1) for i in range(num_nf_rv)), list(min(i//rv_split,1) for i in reversed(range(num_nf_rv)))] * num_flow_layers).astype(np.float32)

def get_old_masks(num_nf_rv, num_flow_layers):
    rv_split = math.ceil(num_nf_rv/2)
    return np.array([list(i//rv_split for i in range(num_nf_rv)), list(i//rv_split for i in reversed(range(num_nf_rv)))] * num_flow_layers).astype(np.float32)


class flow_head(nn.Module):
    def __init__(
            self,
            uncert_type,
            num_flow_layers,
            mask_params_id,
            nflow_mask_type,
            exclude_uncert_idx,
            num_nf_rv,
            cond_nflow,
            in_context_dim,
            out_context_dim,
    ):
        super (flow_head, self).__init__()

        self.uncert_type = uncert_type
        self.cond_nflow = cond_nflow
        self.num_flow_layers = num_flow_layers
        mask_params_id = [int(x) for x in mask_params_id.split('-') if len(x) > 0]
        self.mask_params_id = mask_params_id
        self.out_context_dim = out_context_dim

        self.sel_uncert_part = [x for x in range(24) if str(x) not in exclude_uncert_idx]

        if len(mask_params_id) > 0 and len(self.sel_uncert_part) == 24:
            mask_params = torch.ones(24).float()
            mask_params[torch.LongTensor(mask_params_id)] = 0
            self.register_buffer('mask_params', mask_params)
        self.num_nf_rv = num_nf_rv

        if 'pose' in self.uncert_type:
            prior = distributions.MultivariateNormal(torch.zeros(num_nf_rv), torch.eye(num_nf_rv))
            masks = eval(f'get_{nflow_mask_type}_masks')(num_nf_rv, self.num_flow_layers)
            masks = torch.from_numpy(masks)
        else:
            logger.error(f'Normalizing flow for {self.uncert_type} is not defined')
            exit()

        if self.cond_nflow:
            self.cond_layer = torch.nn.Linear(in_context_dim, self.out_context_dim)
        else:
            out_context_dim = 0

        flow_arch = [num_nf_rv + out_context_dim, 64, num_nf_rv]
        self.flow = RealNVP(net_s, net_t, flow_arch, masks, prior)

    def forward(self, head_output, pred, batch):

        batch_size = pred['smpl_vertices'].shape[0]
        device = pred['smpl_vertices'].device
        context_feats = self.cond_layer(head_output['uncert_feat']) if self.cond_nflow else None

        # During Training
        if 'is_train' in batch.keys():

            if 'pose' in self.uncert_type:
                has_smpl = batch['has_smpl'].bool()
                gt_pose = batch_rodrigues(batch['pose'].view(-1,3)).view(-1,24,3,3)[has_smpl == 1]
                gt_pose = gt_pose[:,self.sel_uncert_part,:]
                pred_pose = pred['pred_pose'][has_smpl == 1][:,self.sel_uncert_part,:]
                pred_sigma = pred['var_pose'][has_smpl == 1]

                if self.cond_nflow:
                    context_feats = context_feats[has_smpl == 1]
                pred_sigma_shape = pred_sigma.shape

                if len(pred_sigma.shape) == 2:
                    pred_sigma = pred_sigma.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3,3)

                bar_pose = torch.abs(pred_pose - gt_pose) / (pred_sigma + 1e-9)
                if self.num_nf_rv == 24:
                    bar_pose = bar_pose.mean(-1).mean(-1)

                bar_pose = bar_pose.reshape(-1, self.num_nf_rv)
                if self.cond_nflow:
                    rp = bar_pose.shape[0] // pred_pose.shape[0]
                    context_feats = torch.repeat_interleave(context_feats, repeats=rp, dim=0)

                log_phi = self.flow.log_prob(bar_pose, context_feats)

                log_phi = log_phi.reshape(pred_pose.shape[0], -1)
                expand_dim = len(pred_sigma_shape) - len(log_phi.shape)

                # x_bar = self.flow.sample(batch_size, context_feats)[has_smpl == 1]

                if log_phi.shape[1] == 24 and len(self.mask_params_id) > 0 and len(self.sel_uncert_part) == 24:
                    log_phi = log_phi * self.mask_params

                while expand_dim > 0:
                    log_phi = log_phi.unsqueeze(-1)
                    expand_dim -= 1

            else:
                logger.error(f'Normalizing flow loss not defined for {self.uncert_type}')
                exit()

        # During Inference
        else:
            log_phi = None

        pred.update({
            'log_phi': log_phi,
            # 'var_scaled': var_scaled
        })
        return pred

