"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import torch

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def get_jnts_from_mesh(pred_vertices, J_regressor, dataset):

    from ..core import constants
    joint_mapper_h36m = constants.H36M_TO_J17 if dataset == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

    # Get 14 predicted joints from the mesh
    pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
    pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
    pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
    pred_keypoints_3d_nonrel = pred_keypoints_3d.clone()
    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

    return pred_keypoints_3d, pred_keypoints_3d_nonrel

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def pampjpe_error(S1, S2, reduction='mean'):

    S1, S2 = S1.cpu().numpy(), S2.cpu().numpy()

    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)

    re_per_joint = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1))
    re = re_per_joint.mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re, re_per_joint

def mpjpe_error(pred_keypoints_3d, gt_keypoints_3d):
    error_per_joint = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
    error = error_per_joint.mean(-1)
    return error, error_per_joint

def vert_error(pred_verts, target_verts=None):
    """
    Args:
        verts_gt (Nx6890x3).
        verts_pred (Nx6890x3).
    Returns:
        error_verts (N).
    """

    if target_verts is None:
        return np.zeros((pred_verts.shape[0]))

    assert len(pred_verts) == len(target_verts)
    v2v = torch.sqrt(((target_verts - pred_verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    return v2v

def conf_oks_correlation(pred_keypoints_3d, gt_keypoints_3d, conf=None, scale=None):

    if conf is None:
        return np.array([0.])
    batch_size = pred_keypoints_3d.shape[0]
    num_joints = pred_keypoints_3d.shape[1]
    num_samples = batch_size * num_joints

    pred_keypoints_3d = pred_keypoints_3d.view(-1, 3).cpu().numpy()
    gt_keypoints_3d = gt_keypoints_3d.view(-1, 3).cpu().numpy()

    from ..utils.kp_utils import get_common_joint_kappas
    # kappa = np.array(get_common_joint_kappas()).astype(np.float32)
    kappa = np.ones(14).astype(np.float32)
    kappa = np.tile(kappa, batch_size)
    # As the images are centered, object scale is 1
    if scale is None:
        scale = torch.ones(batch_size).float()
    scale = torch.repeat_interleave(scale, repeats=num_joints).cpu().numpy()

    # Find Object Keypoint Similarity
    # dist = np.linalg.norm(pred_keypoints_3d - gt_keypoints_3d, axis=-1)
    criterion = torch.nn.MSELoss(reduction='none')
    dist = criterion(pred_keypoints_3d, gt_keypoints_3d).mean(-1).reshape(-1).cpu().numpy()

    oks = np.exp(-(dist**2) / (2 * (scale**2) * (kappa**2))) / num_samples

    # Find confidence of common joints
    from ..core.constants import SMPL_J24_TO_COMMON_J14
    conf = conf[:, SMPL_J24_TO_COMMON_J14].flatten()

    return oks, conf


def calculate_distance_pose(pred_pose, gt_pose):
    criterion = torch.nn.MSELoss(reduction='none')
    from .geometry import batch_rodrigues

    gt_pose = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)
    dist = criterion(pred_pose, gt_pose).mean(-1).mean(-1)
    return dist

def calculate_pearson_coff(x, y):
    from scipy import stats
    r, p = stats.pearsonr(x, y)
    return np.array([r]), np.array([p])






def reconstruction_errori_old(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -math.inf
        self.min = math.inf

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)
