import cv2
import os
import time
import torch
import numpy as np
from os.path import join
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..core import constants
from ..core.config import DATASET_FILES, DATASET_FOLDERS, SMPL_MODEL_DIR,  \
                          JOINT_REGRESSOR_H36M, DATASET_NPZ_PATH, PASCAL_ROOT
from ..utils.image_utils import crop_cv2, flip_img, flip_pose, flip_kp, transform, rot_aa, \
                                calculate_bbox_info, calculate_focal_length, \
                                convert_crop_coords_to_orig_img_cliff
from .occlusion import load_pascal_occluders, occlude_with_pascal_objects_kp
from ..utils.train_utils import get_confident_frames
from ..models import SMPL


# Re-intiialize cuda in forked processes for processing visible vertices
# Where tensors needs to be in cuda for processing
'''
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
'''

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, method, dataset, ignore_3d=False, use_augmentation=True,
                 is_train=True, num_images=0, img_upscale=4, debug=False):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.method = method
        self.img_upscale = img_upscale
        self.debug = debug
        self.img_dir = join(options.DATA_DIR, DATASET_FOLDERS[dataset])
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        ds_file = join(options.DATA_DIR, DATASET_NPZ_PATH, options.DATA_TYPE, DATASET_FILES[is_train][dataset])
        logger.info(f'Loading npz file from {ds_file}...')
        self.data = np.load(ds_file)
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
        self.imgname = self.data['imgname']


        self.dataset_dict = {dataset: 0}

        # If the dataset is inferred from POCO, select confident frames
        try:
            var = self.data['var']
            conf_select_idx = get_confident_frames(var, options.UNCERT_THRESHOLD)
            if num_images > 0:
                rand = np.random.randint(0, conf_select_idx.shape[0], size=(num_images))
                conf_select_idx = conf_select_idx[rand]
            self.imgname = self.imgname[conf_select_idx]
            self.data_subset = {}
            for f in self.data.files:
                self.data_subset[f] = self.data[f][conf_select_idx]
            self.data = self.data_subset
        except KeyError:
            if num_images > 0:
                # select a random subset of the dataset
                rand = np.random.randint(0, len(self.imgname), size=(num_images))
                logger.info(f'{rand.shape[0]} images are randomly sampled from {self.dataset}')
                self.imgname = self.imgname[rand]
                self.data_subset = {}
                self.grphnames_subset = {}
                for f in self.data.files:
                    self.data_subset[f] = self.data[f][rand]
                self.data = self.data_subset

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))

        # SMPL poses for SPIN multi-view mpi-inf-3dhp are not correct, hence don't use them
        # Use SMPL fits from SPIN
        if dataset == 'mpi-inf-3dhp' and self.has_smpl.nonzero()[0].shape[0] < len(self.imgname):
            self.has_smpl = np.zeros(len(self.imgname))

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

        # Get Person ID data if available
        try:
            self.person_id = self.data['person_id']
        except KeyError:
            self.person_id = -1*np.ones(len(self.imgname)).astype(np.int32)

        self.occluders = None
        if is_train and self.options.USE_SYNTHETIC_OCCLUSION:
            if self.options.OCC_AUG_DATASET == 'pascal':
                self.occluders = load_pascal_occluders(pascal_voc_root_path=PASCAL_ROOT)
                logger.info(f'Found {len(self.occluders)} suitable '
                            f'objects from {self.options.OCC_AUG_DATASET} dataset')

        # evaluation variables
        if not self.is_train:
            self.joint_mapper_h36m = constants.H36M_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.H36M_TO_J14
            self.joint_mapper_gt = constants.J24_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.J24_TO_J14
            self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()

            self.smpl_male = SMPL(SMPL_MODEL_DIR, gender='male', create_transl=False)
            self.smpl_female = SMPL(SMPL_MODEL_DIR, gender='female', create_transl=False)

        self.length = self.scale.shape[0]
        logger.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train and self.use_augmentation == True:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5 and self.options.FLIP:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.NOISE_FACTOR, 1+self.options.NOISE_FACTOR, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.ROT_FACTOR,
                    max(-2*self.options.ROT_FACTOR, np.random.randn()*self.options.ROT_FACTOR))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.SCALE_FACTOR,
                    max(1-self.options.SCALE_FACTOR, np.random.randn()*self.options.SCALE_FACTOR+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, kp2d=None, \
                       occ_aug=False, img_upscale=1):
        """Process rgb image and do augmentation."""
        rgb_img = crop_cv2(rgb_img, center, scale,
                  [img_upscale*self.options.IMG_RES, img_upscale*self.options.IMG_RES], rot=rot)
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)

        if self.occluders is not None and occ_aug and self.dataset in ['h36m', 'coco', '3dpw']:
            if self.options.OCC_AUG_DATASET == 'pascal':
                # rgb_img = occlude_with_pascal_objects(rgb_img, self.occluders)
                rgb_img = occlude_with_pascal_objects_kp(rgb_img, kp2d.copy(), scale, self.occluders)

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2] + 1, center, scale,
                                  [self.options.IMG_RES, self.options.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2. *kp[:,:-1] / self.options.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        keypoints_orig = self.keypoints[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except:
            logger.info(f'Cannot access {imgname}')
            rnd_sleep = np.random.randint(500, 1000)
            logger.info(f'Sleeping for {rnd_sleep} seconds to avoid access issues')
            time.sleep(rnd_sleep)
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.j2d_processing(keypoints, center, sc * scale, rot, flip)

        # Get 2D keypoints in orig image for CLIFF
        bbox = np.array([[center[0], center[1], sc*scale*200.]])
        keypoints_cliff = convert_crop_coords_to_orig_img_cliff(bbox, \
                          keypoints[None, ...].copy(), self.options.IMG_RES)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn, \
                                  kp2d=keypoints, occ_aug=self.options.USE_SYNTHETIC_OCCLUSION)
        img = torch.from_numpy(img).float()

        # Store image before normalization to use it in visualization
        item['index'] = index
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        item['keypoints'] = torch.from_numpy(keypoints).float()
        item['keypoints_orig'] = torch.from_numpy(keypoints_orig).float()
        item['keypoints_fullimg'] = torch.from_numpy(keypoints_cliff).float()

        item['focal_length'] = calculate_focal_length(orig_shape[0], orig_shape[1])
        item['bbox_info'] = calculate_bbox_info(center, sc*scale, orig_shape)

        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['person_id'] = self.person_id[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        # prepare pose_3d for evaluation
        # For 3DPW get the 14 common joints from the rendered shape
        if not self.is_train:
            if self.dataset in ['3dpw', '3dpw-all']:
                if self.options.GENDER_EVAL == True:
                    gt_vertices = self.smpl_male(global_orient=item['pose'].unsqueeze(0)[:,:3],
                                                 body_pose=item['pose'].unsqueeze(0)[:,3:],
                                                 betas=item['betas'].unsqueeze(0)).vertices
                    gt_vertices_f = self.smpl_female(global_orient=item['pose'].unsqueeze(0)[:,:3],
                                                     body_pose=item['pose'].unsqueeze(0)[:,3:],
                                                     betas=item['betas'].unsqueeze(0)).vertices
                    gt_vertices = gt_vertices if item['gender'] == 0 else gt_vertices_f
                else:
                    gt_vertices = self.smpl(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0),
                    ).vertices

                J_regressor_batch = self.J_regressor[None, :].expand(1, -1, -1)
                pose_3d = torch.matmul(J_regressor_batch, gt_vertices)
                pelvis = pose_3d[:, [0], :].clone()
                pose_3d = pose_3d[:, self.joint_mapper_h36m, :]
                item['pose_3d_nonrel'] = pose_3d[0].float()
                pose_3d = pose_3d - pelvis
                item['pose_3d'] = pose_3d[0].float()
                item['vertices'] = gt_vertices[0].float()

                # full pose_3d for 3DPW
                gt_jnts = self.smpl(
                    global_orient=item['pose'].unsqueeze(0)[:, :3],
                    body_pose=item['pose'].unsqueeze(0)[:, 3:],
                    betas=item['betas'].unsqueeze(0)).joints[0, 25:, :].float()
                gt_jnts = torch.hstack((gt_jnts, torch.ones((gt_jnts.shape[0], 1))))
                item['pose_3d_full'] = gt_jnts

                
            else:
                item['pose_3d_full'] = item['pose_3d'].float()
                item['pose_3d'] = item['pose_3d'][self.joint_mapper_gt, :-1].float()
        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        if self.is_train:
            item['is_train'] = 1.

        return item

    def __len__(self):
        return len(self.imgname)
