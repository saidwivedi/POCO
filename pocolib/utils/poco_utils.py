import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from .kp_utils import get_smpl_joint_names, get_smpl_skeleton
from .train_utils import is_main_process
from .eval_utils import AverageMeter as AvgFn
from .geometry import batch_rodrigues, matrix_to_rotation_6d
from ..losses.losses import smpl_err, keypoint_3d_loss

def get_avg(uncert_dict):
    return {key: val.avg for key, val in uncert_dict.items()}

def get_max(uncert_dict):
    return {key: val.max for key, val in uncert_dict.items()}

def get_min(uncert_dict):
    return {key: val.min for key, val in uncert_dict.items()}

def get_kinematic_uncert(var):
    smpl_kinematic = get_smpl_skeleton()
    for i in smpl_kinematic[:,1]:
        var[:,i] += var[:,smpl_kinematic[i-1,0]]
    return var

class POCOUtils():

    def __init__(self, hparams):
        super(POCOUtils, self).__init__()
        self.log_uncert_count = 0
        self.PL_LOGGING = hparams.PL_LOGGING
        self.pref_logger = hparams.PREF_LOGGER
        self.METHOD = hparams.METHOD
        self.LOSS_VER = hparams.POCO.LOSS_VER
        self.HPS_BACKBONE = hparams.POCO.BACKBONE
        self.UNCERT_TYPE = hparams.POCO.UNCERT_TYPE
        self.KINEMATIC_UNCERT = hparams.POCO.KINEMATIC_UNCERT
        self.LOG_UNCERT_STAT = hparams.POCO.LOG_UNCERT_STAT
        self.sel_uncert_part = [x for x in range(24) if str(x) not in hparams.POCO.EXCLUDE_UNCERT_IDX.split('-')]
        self.smpl_pose_names = np.array(get_smpl_joint_names())[self.sel_uncert_part].tolist()
        self.smpl_beta_names = [f'beta_{i}' for i in range(10)]
        if is_main_process() and self.METHOD == 'poco':
            self.init_uncert_variables('tr')
            self.init_uncert_variables('val')

        if not isinstance(self.UNCERT_TYPE, list):
            self.UNCERT_TYPE = [self.UNCERT_TYPE]

    def get_global_uncert(self, var, sensitivity_threshold=0.40):

        if 'cliff' in self.HPS_BACKBONE:
            var[var[:,0] > 2*sensitivity_threshold] = 1.0
            global_var = var
            global_var = global_var[:,0]
        elif 'pare' in self.HPS_BACKBONE:
            var[var[:,0] > sensitivity_threshold] = 1.0
            global_var = var[:].mean(-1)

        return global_var

    def prepare_uncert(self, var, return_torch=False, return_conf=False):

        if isinstance(var, torch.Tensor):
            var = var.detach()

        if len(var.shape) == 4:
            var = var.mean(-1).mean(-1)
        elif len(var.shape) == 3:
            var = var.mean(-1)
        if self.LOSS_VER == 'gauss_logsigma':
            var = np.exp(var)
        elif self.LOSS_VER == 'delta':
            var_size = var.shape[1] // 2
            alpha, gamma = var[:,:var_size], var[:,var_size:]
            var = alpha / (gamma ** 2)
        elif self.LOSS_VER == 'genG' or self.LOSS_VER == 'mse_genG':
            var_size = var.shape[1] // 2
            alpha, beta = var[:,:var_size], var[:,var_size:]
            pred_uncert1 = (alpha**2)*(torch.exp(torch.lgamma(3/(beta + 1e-6))))
            pred_uncert2 = torch.exp(torch.lgamma(1/(beta + 1e-6)))
            var = pred_uncert1 / pred_uncert2
        elif self.LOSS_VER == 'gauss_genG':
            var_size = var.shape[1] // 3
            var = var[:, :var_size]

        if isinstance(var, torch.Tensor) and not return_torch:
            var = var.cpu().numpy()
        if self.KINEMATIC_UNCERT:
            var = get_kinematic_uncert(var)
        if return_conf:
            var = 1 - var

        return var


    def init_uncert_variables(self, dtype):
        uncert_type = self.UNCERT_TYPE
        pnames, bnames = self.smpl_pose_names, self.smpl_beta_names
        plen, blen = len(pnames), len(bnames)

        if 'pose' in uncert_type:
            exec(f'self.{dtype}_uncert_pose = dict([({pnames}[i], []) for i in range({plen})])')
            exec(f'self.{dtype}_pose_err = dict([({pnames}[i], []) for i in range({plen})])')

    def reset_uncert_stats(self, dtype):
        uncert_type = self.UNCERT_TYPE
        pnames, bnames = self.smpl_pose_names, self.smpl_beta_names
        plen, blen = len(pnames), len(bnames)
        if 'pose' in uncert_type:
            exec(f'self.{dtype}_pose_min = dict([({pnames}[i], AvgFn()) for i in range({plen})])')
            exec(f'self.{dtype}_pose_max = dict([({pnames}[i], AvgFn()) for i in range({plen})])')
            exec(f'self.{dtype}_pose_mean = dict([({pnames}[i], AvgFn()) for i in range({plen})])')

    # Error Analysis
    def get_smpl_err(self, pred, batch):
        pred_pose = pred['pred_pose']
        if 'gauss' in self.LOSS_VER or 'genG' in self.LOSS_VER: # Pose - BxJx6x6
            gt_pose = batch_rodrigues(batch['pose'].view(-1,3)).view(-1, 24, 3, 3)
            criterion = nn.MSELoss(reduction='none')
        elif 'norm_flow' in self.LOSS_VER: # Pose - BxJx3x3
            gt_pose = batch_rodrigues(batch['pose'].view(-1,3)).view(-1, 24, 3, 3)
            criterion = lambda x,y: x-y
        err_pose, err_betas = smpl_err(pred_pose.detach(),
                                       pred['pred_shape'].detach(),
                                       gt_pose,
                                       batch['betas'],
                                       batch['has_smpl'].bool(),
                                       criterion)
        err_pose, err_betas = err_pose.cpu().numpy(), err_betas.cpu().numpy()
        if len(err_pose.shape) == 4: # BxJx3x3
            err_pose = err_pose.mean(-1).mean(-1)
        elif len(err_pose.shape) == 3: # BxJx6
            err_pose = err_pose.mean(-1)
        return err_pose, err_betas

    def get_jnt_err(self, pred, batch):
        gt_joints = batch['pose_3d_full'] if 'pose_3d_full' in batch.keys() else batch['pose_3d']
        pred_joints = pred['smpl_joints3d'].detach()
        has_pose_3d = batch['has_pose_3d'].bool()
        has_pose_3d = has_pose_3d if torch.any(has_pose_3d) else batch['has_smpl'].bool()

        err_joints = keypoint_3d_loss(pred_joints,
                                      gt_joints,
                                      has_pose_3d,
                                      reduction=False)
        err_joints = err_joints.cpu().numpy()
        return err_joints


    def accumulate_pose_uncert(self, dtype, pred, err_pose):

        pred_uncert = self.prepare_uncert(pred['var_pose'])
        # if 'norm_flow' in self.LOSS_VER:
        #     pred_scaled_uncert = self.prepare_uncert(pred['var_scaled'])

        # Accumulating Error and Uncertainty
        for i, key in enumerate(self.smpl_pose_names):
            # if 'norm_flow' in self.LOSS_VER:
            #     eval(f'self.{dtype}_scaled_pose_min[key]').update(pred_scaled_uncert[:,i].min())
            #     eval(f'self.{dtype}_scaled_pose_max[key]').update(pred_scaled_uncert[:,i].max())
            #     eval(f'self.{dtype}_scaled_uncert_pose[key]').extend(pred_scaled_uncert[:,i])
            eval(f'self.{dtype}_uncert_pose[key]').extend(pred_uncert[:,i])
            eval(f'self.{dtype}_pose_err[key]').extend(err_pose[:,i])
            eval(f'self.{dtype}_pose_min[key]').update(pred_uncert[:,i].min())
            eval(f'self.{dtype}_pose_max[key]').update(pred_uncert[:,i].max())
            eval(f'self.{dtype}_pose_mean[key]').update(pred_uncert[:,i].mean())

        return pred_uncert.mean(1)

    def accumulate_betas_uncert(self, dtype, pred, err_betas):
        if self.LOSS_VER == 'gauss_logsigma':
            pred_uncert = pred['var_betas'].detach().exp().cpu().numpy()
        elif self.LOSS_VER == 'gauss_sigma':
            pred_uncert = pred['var_betas'].detach().cpu().numpy()
        elif self.LOSS_VER == 'delta':
            var_betas = pred['var_betas'].detach().cpu().numpy()
            var_size = var_betas.shape[1] // 2
            alpha, gamma = var_betas[:,:var_size], var_betas[:,var_size:]
            pred_uncert = alpha / (gamma ** 2)
        elif 'norm_flow' in self.LOSS_VER:
            print('Not Implemented')

        # Accumulating Error and Uncertainty
        for i, key in enumerate(self.smpl_beta_names):
            eval(f'self.{dtype}_uncert_betas[key]').extend(pred_uncert[:,i])
            eval(f'self.{dtype}_betas_err[key]').extend(err_betas[:,i])
            eval(f'self.{dtype}_betas_min[key]').update(pred_uncert[:,i].min())
            eval(f'self.{dtype}_betas_max[key]').update(pred_uncert[:,i].max())
            eval(f'self.{dtype}_betas_mean[key]').update(pred_uncert[:,i].mean())

        return pred_uncert.mean(1)

    def accumulate_joints_uncert(self, dtype, pred, err_joints):

        pred_uncert = self.prepare_uncert(pred['var_joints'])
        # if 'norm_flow' in self.LOSS_VER:
        #     pred_scaled_uncert = self.prepare_uncert(pred['var_scaled'])

        # Accumulating Error and Uncertainty
        for i, key in enumerate(self.smpl_pose_names):
            # if 'norm_flow' in self.LOSS_VER:
            #     eval(f'self.{dtype}_scaled_joints_min[key]').update(pred_scaled_uncert[:,i].min())
            #     eval(f'self.{dtype}_scaled_joints_max[key]').update(pred_scaled_uncert[:,i].max())
            #     eval(f'self.{dtype}_scaled_uncert_joints[key]').extend(pred_scaled_uncert[:,i])
            eval(f'self.{dtype}_uncert_joints[key]').extend(pred_uncert[:,i])
            eval(f'self.{dtype}_joints_err[key]').extend(err_joints[:,i].mean(1))
            eval(f'self.{dtype}_joints_min[key]').update(pred_uncert[:,i].min())
            eval(f'self.{dtype}_joints_max[key]').update(pred_uncert[:,i].max())
            eval(f'self.{dtype}_joints_mean[key]').update(pred_uncert[:,i].mean())

        return pred_uncert.mean(1)


    def send_hist3d_to_comet(self, values, name, step, online_logger):
        online_logger.experiment.log_histogram_3d(values=values, name=name, step=step)

    def send_hist3d_to_tensorboard(self, values, name, step, online_logger):
        online_logger.experiment.add_histogram(name, np.array(values), step)


    def log_pose_uncert(self, online_logger, dtype, step):

        # Log individual param uncertainty and SMPL Error
        for i, key in enumerate(self.smpl_pose_names):
            for j, pr_log in enumerate(self.pref_logger):
                eval(f'self.send_hist3d_to_{pr_log}')(
                                        values=eval(f'self.{dtype}_uncert_pose[key]'),
                                        name=f'{dtype}/uncert_pose_{i:02d}_{key}',
                                        step=step,
                                        online_logger=online_logger[j])
                eval(f'self.send_hist3d_to_{pr_log}')(
                                        values=eval(f'self.{dtype}_pose_err[key]'),
                                        name=f'{dtype}/err_pose_{i:02d}_{key}',
                                        step=step,
                                        online_logger=online_logger[j])

    def log_uncert_stats(self, online_logger, dtype, un_type, step):

        for limit in ['min', 'max', 'mean']:
            data = get_avg(eval(f'self.{dtype}_{un_type}_{limit}'))
            uncert_list = sorted(data.items())
            key, uncert_values = zip(*uncert_list)
            plt.bar(key, uncert_values)
            plt.xticks(rotation=90)
            plt.tight_layout()
            online_logger[0].experiment.add_figure(f'stats_{dtype}_{un_type}_{limit}', plt.gcf(), step)
            plt.clf()
        plt.close('all')

    def get_uncert_stats(self):
        uncert_stats = {}
        for dtype in ['tr', 'val']:
            for un_type in self.UNCERT_TYPE:
                if f'{dtype}_{un_type}_min' in vars(self).keys():
                    uncert_stats[f'{dtype}_{un_type}_min'] = get_min(eval(f'self.{dtype}_{un_type}_min'))
                    uncert_stats[f'{dtype}_{un_type}_min_avg'] = get_avg(eval(f'self.{dtype}_{un_type}_min'))
                if f'{dtype}_{un_type}_max' in vars(self).keys():
                    uncert_stats[f'{dtype}_{un_type}_max'] = get_max(eval(f'self.{dtype}_{un_type}_max'))
                    uncert_stats[f'{dtype}_{un_type}_max_avg'] = get_avg(eval(f'self.{dtype}_{un_type}_max'))
                if f'{dtype}_{un_type}_mean' in vars(self).keys():
                    uncert_stats[f'{dtype}_{un_type}_mean'] = get_avg(eval(f'self.{dtype}_{un_type}_mean'))
        return uncert_stats

    def accumulate_uncert(self, dtype, pred, batch, batch_nb):

        # from inspect import stack
        # logger.info("CALLER FUNCTION: {}".format(stack()[1].function))

        if 'pose' in self.UNCERT_TYPE or 'betas' in self.UNCERT_TYPE:
            err_pose, err_betas = self.get_smpl_err(pred, batch)

        if batch_nb == 0:
            self.reset_uncert_stats(dtype)

        total_var = np.zeros(pred['pred_pose'].shape[0])
        for un_type in self.UNCERT_TYPE:
            err = eval(f'err_{un_type}')
            total_var += eval(f'self.accumulate_{un_type}_uncert')(dtype, pred, err)

        return total_var

    def log_uncert(self, dtype, online_logger, step):

        # for un_type in self.UNCERT_TYPE:
        #     eval(f'self.log_{un_type}_uncert')(online_logger, dtype, step)
            # TODO: Stop logging the uncert stats for now
            # if dtype == 'tr':
            #     self.log_uncert_count += 1
            #     if self.log_uncert_count % self.LOG_UNCERT_STAT == 0:
            #         self.log_uncert_stats(online_logger, dtype, un_type, step)
            # elif dtype == 'val':
            #     self.log_uncert_stats(online_logger, dtype, un_type, step)
        self.init_uncert_variables(dtype)
