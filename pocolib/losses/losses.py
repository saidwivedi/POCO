import torch
import math
import numpy as np
import torch.nn as nn

from loguru import logger
from ..models import SMPL
from ..core import constants
from ..core.config import SMPL_MODEL_DIR
from ..utils.geometry import batch_rodrigues

from ..utils.image_utils import show_imgs
from .segmentation import CrossEntropy

class HMRLoss(nn.Module):
    def __init__(
            self,
            shape_loss_weight=0,
            keypoint3d_loss_weight=5.,
            keypoint2d_loss_weight=2.5,
            keypoint2d_noncrop=False,
            pose_loss_weight=1.,
            beta_loss_weight=0.001,
            openpose_train_weight=0.,
            gt_train_weight=1.,
            loss_weight=60.,
            use_smpl_render_loss=False,
            use_smpl_segm_loss=False,
            smpl_render_loss_weight=1.,
            smpl_segm_loss_weight=1.,
    ):
        super(HMRLoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()

        self.criterion_segm_mask = CrossEntropy()
        self.criterion_render = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint3d_loss_weight = keypoint3d_loss_weight
        self.keypoint2d_loss_weight = keypoint2d_loss_weight
        self.openpose_train_weight = openpose_train_weight

        self.use_smpl_render_loss = use_smpl_render_loss
        self.smpl_render_loss_weight = smpl_render_loss_weight
        self.use_smpl_segm_loss = use_smpl_segm_loss
        self.smpl_segm_loss_weight = smpl_segm_loss_weight
        self.keypoint2d_noncrop = keypoint2d_noncrop

    def forward(self, pred, gt, cur_iteration):
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        img_size = gt['orig_shape'].rot90().T.unsqueeze(1)
        gt_pose = gt['pose']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()

        # Use full image keypoints
        if self.keypoint2d_noncrop:
            # normalize predicted keypoints between -1 and 1 to compute the loss
            gt_keypoints_2d = gt['keypoints_fullimg'].clone()
            pred_projected_keypoints_2d[:, :, :2] = 2 * (pred_projected_keypoints_2d[:, :, :2] / img_size) - 1
            gt_keypoints_2d[:, :, :2] = 2 * (gt_keypoints_2d[:, :, :2] / img_size) - 1
        else:
            # Use crop keypoints
            gt_keypoints_2d = gt['keypoints']

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        if self.keypoint2d_noncrop:
            loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] * 200.).unsqueeze(-1)
            loss_keypoints = loss_keypoints * loss_keypoints_scale.unsqueeze(1)
            loss_keypoints = loss_keypoints.mean()
        else:
            loss_keypoints = loss_keypoints.mean()

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )


        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint2d_loss_weight
        loss_keypoints_3d *= self.keypoint3d_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean() * 0.016

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }

        if self.use_smpl_render_loss:
            loss_smpl_render = self.criterion_render(pred['pred_smpl_render'], gt['gt_smpl_render'])
            loss_smpl_render *= self.smpl_render_loss_weight
            loss_dict['loss/loss_smpl_render'] = loss_smpl_render

        if self.use_smpl_segm_loss:
            pred_segm_mask = pred['pred_segm_mask'][has_smpl == 1]
            gt_segm_mask = gt['gt_segm_mask'][has_smpl == 1]
            loss_smpl_segm = self.criterion_segm_mask(score=pred_segm_mask, target=gt_segm_mask)
            loss_smpl_segm *= self.smpl_segm_loss_weight
            loss_dict['loss/loss_smpl_segm'] = loss_smpl_segm

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        # if loss > 100:
        #     print('Loss dict -> ', loss_dict)
        #     print('Image names -> ', gt['imgname'])

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict

class POCOLoss(nn.Module):
    def __init__(
            self,
            shape_loss_weight=0,
            keypoint3d_loss_weight=5.,
            keypoint2d_loss_weight=2.5,
            keypoint2d_noncrop=False,
            pose_loss_weight=1.,
            beta_loss_weight=0.001,
            openpose_train_weight=0.,
            gt_train_weight=1.,
            pose_uncert_weight=1.,
            beta_uncert_weight=1.,
            jnt_uncert_weight=1.,
            nf_loss_weight=1.,
            genG_loss_weight=1.,
            use_keyconf=False,
            loss_weight=60.,
            loss_ver='norm_flow_res_gaus',
            uncert_type=['pose'],
            exclude_uncert_idx='',
            use_smpl_render_loss=False,
            use_smpl_segm_loss=False,
            smpl_render_loss_weight=1.,
            smpl_segm_loss_weight=1.,
    ):
        super(POCOLoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()

        self.criterion_segm_mask = CrossEntropy()
        self.criterion_render = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint3d_loss_weight = keypoint3d_loss_weight
        self.keypoint2d_loss_weight = keypoint2d_loss_weight
        self.openpose_train_weight = openpose_train_weight
        self.pose_uncert_weight = pose_uncert_weight
        self.beta_uncert_weight = beta_uncert_weight
        self.jnt_uncert_weight = jnt_uncert_weight
        self.nf_loss_weight = nf_loss_weight
        self.genG_loss_weight = genG_loss_weight
        self.use_keyconf = use_keyconf
        self.loss_ver = loss_ver
        self.uncert_type = uncert_type

        self.sel_uncert_part = [x for x in range(24) if str(x) not in exclude_uncert_idx.split('-')]

        # Uncert weight multiplier
        if 'pose' in self.uncert_type and self.loss_ver != 'norm_flow':
            self.pose_loss_weight *= self.pose_uncert_weight

        self.use_smpl_render_loss = use_smpl_render_loss
        self.smpl_render_loss_weight = smpl_render_loss_weight
        self.use_smpl_segm_loss = use_smpl_segm_loss
        self.smpl_segm_loss_weight = smpl_segm_loss_weight
        self.keypoint2d_noncrop = keypoint2d_noncrop

    def forward(self, pred, gt, cur_iteration):

        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        batch_size = pred_joints.shape[0]

        empty_tensor = torch.empty(batch_size)
        pred_uncert_pose = pred['var_pose'] if 'var_pose' in pred.keys() else empty_tensor
        pred_uncert_joints = pred['var_joints'] if 'var_joints' in pred.keys() else empty_tensor
        pred_uncert_betas = pred['var_betas'] if 'var_betas' in pred.keys() else empty_tensor
        gt_pose_cond_idx = pred['gt_pose_cond_idx']

        img_size = gt['orig_shape'].rot90().T.unsqueeze(1)
        gt_pose = gt['pose']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()

        # Use full image keypoints
        if self.keypoint2d_noncrop:
            # normalize predicted keypoints between -1 and 1 to compute the loss
            pred_projected_keypoints_2d[:, :, :2] = 2 * (pred_projected_keypoints_2d[:, :, :2] / img_size) - 1
            gt_keypoints_2d = gt['keypoints_fullimg'].clone()
            gt_keypoints_2d[:, :, :2] = 2 * (gt_keypoints_2d[:, :, :2] / img_size) - 1
        else:
            # Use crop keypoints
            gt_keypoints_2d = gt['keypoints']


        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses_uncertainty(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            gt_pose_cond_idx,
            pred_uncert_pose,
            pred_uncert_betas,
            self.loss_ver,
            self.uncert_type,
            self.sel_uncert_part,
            self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        if self.keypoint2d_noncrop:
            loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] * 200.).unsqueeze(-1)
            loss_keypoints = loss_keypoints * loss_keypoints_scale.unsqueeze(1)
            loss_keypoints = loss_keypoints.mean()
        else:
            loss_keypoints = loss_keypoints.mean()

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )

        # Basic Losses
        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint2d_loss_weight
        loss_keypoints_3d *= self.keypoint3d_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight * self.beta_uncert_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean() * 0.016

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }

        # SMPL Render Loss
        if self.use_smpl_render_loss:
            loss_smpl_render = self.criterion_render(pred['pred_smpl_render'], gt['gt_smpl_render'])
            loss_smpl_render *= self.smpl_render_loss_weight
            loss_dict['loss/loss_smpl_render'] = loss_smpl_render

        # SMPL Part Segmentation Loss
        if self.use_smpl_segm_loss:
            pred_segm_mask = pred['pred_segm_mask'][has_smpl == 1]
            gt_segm_mask = gt['gt_segm_mask'][has_smpl == 1]
            loss_smpl_segm = self.criterion_segm_mask(score=pred_segm_mask, target=gt_segm_mask)
            loss_smpl_segm *= self.smpl_segm_loss_weight
            loss_dict['loss/loss_smpl_segm'] = loss_smpl_segm

        # Normalizing flow loss for uncertainty estimate
        log_phi = pred['log_phi'] if 'log_phi' in pred.keys() else torch.FloatTensor(0)
        if log_phi.nelement() > 0:
            if 'pose' in self.uncert_type:
                loss_nf = (torch.log(pred_uncert_pose[has_smpl == 1]) - log_phi).mean() * self.nf_loss_weight
            loss_dict['loss/loss_nf'] = loss_nf

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        if loss > 50 or torch.isnan(loss): #debug
            print(loss_dict)

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict


def projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        openpose_weight,
        gt_weight,
        criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    conf[:, :25] *= openpose_weight 
    conf[:, 25:] *= gt_weight
    loss = conf * criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])
    return loss


def keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    loss = criterion(pred_keypoints_2d, gt_keypoints_2d).mean()
    return loss


def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        has_pose_3d,
        criterion=None,
        reduction=True,
):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    batch_size = gt_keypoints_3d.shape[0]
    criterion = nn.MSELoss(reduction='none') if criterion is None else criterion
    if gt_keypoints_3d.shape[1] == 24:
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        if reduction:
            return criterion(pred_keypoints_3d, gt_keypoints_3d).mean()
        else:
            return criterion(pred_keypoints_3d, gt_keypoints_3d)
    else:
        return torch.FloatTensor(batch_size).fill_(0.).to(pred_keypoints_3d.device)


def shape_loss(
        pred_vertices,
        gt_vertices,
        has_smpl,
        criterion,
):
    """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)


def smpl_losses_uncertainty(
        pred_rotmat,
        pred_betas,
        gt_pose,
        gt_betas,
        has_smpl,
        gt_pose_cond_idx,
        pred_uncert_pose,
        pred_uncert_betas,
        loss_ver,
        uncert_type,
        sel_uncert_part,
        criterion,
):
    device = gt_pose.device
    batch_size = gt_pose.shape[0]
    gt_rotmat = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)
    uncert_idx = has_smpl
    not_uncert_idx = torch.cuda.BoolTensor(batch_size).fill_(0)
    if len(gt_pose_cond_idx) > 0:
        not_uncert_idx[gt_pose_cond_idx] = 1
        not_uncert_idx = torch.logical_and(not_uncert_idx, has_smpl)
        uncert_idx = torch.cuda.BoolTensor(batch_size).fill_(1)
        uncert_idx[gt_pose_cond_idx] = 0
        uncert_idx = torch.logical_and(uncert_idx, has_smpl)

    pred_rotmat_valid = pred_rotmat[uncert_idx == 1]
    pred_rotmat_no_uncert = pred_rotmat[not_uncert_idx == 1]
    gt_rotmat_valid = gt_rotmat[uncert_idx == 1]
    gt_rotmat_no_uncert = gt_rotmat[not_uncert_idx == 1]

    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    eps = 1e-8

    loss_regr_pose, loss_regr_betas = None, None
    if len(pred_rotmat_valid) > 0:
        # Pose Loss
        if 'pose' in uncert_type:
            pose_var = pred_uncert_pose[uncert_idx == 1]
            if len(pose_var.shape) == 2:
                pose_var = pose_var.unsqueeze(2).unsqueeze(2).repeat(1,1,3,3)
            elif loss_ver == 'norm_flow_res':
                amp = 1 / math.sqrt(2 * math.pi)
                var_loss = torch.log(pose_var / amp)
                pose_loss = torch.abs(pred_rotmat_valid - gt_rotmat_valid)
                logQ = var_loss + (pose_loss / (math.sqrt(2) * pose_var + 1e-9))
                loss_regr_pose = logQ.mean()
            elif loss_ver == 'norm_flow_res_gaus':
                if pose_var.shape[1] < 24: # Some parts are excluded from uncertainty
                    loss_regr_pose = criterion(pred_rotmat_valid, gt_rotmat_valid)
                else:
                    pose_loss1 = torch.pow(pred_rotmat_valid - gt_rotmat_valid, 2) / (pose_var + eps)
                    pose_loss2 = torch.log(pose_var + eps)
                    loss_regr_pose = 0.5 * (pose_loss1 + pose_loss2).mean()
            else:
                loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)

        if loss_regr_pose is None:
            loss_regr_pose = criterion(pred_rotmat_valid, gt_rotmat_valid)
        if loss_regr_betas is None:
            loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)

    gt_var = pred_uncert_pose[not_uncert_idx == 1]
    if len(gt_var) > 0:
        loss_regr_pose_no_uncert = criterion(pred_rotmat_no_uncert, gt_rotmat_no_uncert)
        loss_gt_var = gt_var.mean()
        loss_regr_pose += loss_regr_pose_no_uncert + loss_gt_var

    return loss_regr_pose, loss_regr_betas


def smpl_losses(
        pred_rotmat,
        pred_betas,
        gt_pose,
        gt_betas,
        has_smpl,
        criterion=None,
):
    batch_size = gt_pose.shape[0]
    criterion = nn.MSELoss(reduction='none') if criterion is None else criterion
    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = criterion(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(batch_size).fill_(0.).to(pred_rotmat.device)
        loss_regr_betas = torch.FloatTensor(batch_size).fill_(0.).to(pred_rotmat.device)
    return loss_regr_pose, loss_regr_betas

def smpl_err(
        pred_pose,
        pred_betas,
        gt_pose,
        gt_betas,
        has_smpl,
        criterion,
):
    batch_size = gt_pose.shape[0]
    pred_pose_valid = pred_pose[has_smpl == 1]
    gt_pose_valid = gt_pose[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_pose) > 0:
        err_pose = criterion(pred_pose_valid, gt_pose_valid)
        err_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        err_pose = torch.zeros_like(gt_pose)
        err_betas = torch.zeros_like(gt_betas)
    return err_pose, err_betas


def neg_iou_loss(predict, target):

    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims) + 1e-6
    union = (predict + target - predict * target).sum(dims) + 1e-6
    neg_iou = 1. - (intersect / union).sum() / intersect.nelement()

    return neg_iou

