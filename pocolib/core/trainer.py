import os
import cv2
import json
import math
import torch
import joblib
import numpy as np
from loguru import logger
from os.path import isfile
import pytorch_lightning as pl
from smplx import SMPL as SMPL_native
from torch.utils.data import DataLoader

from . import config
from . import constants
from ..models import SMPL, POCO
from ..losses import POCOLoss
from ..utils.image_utils import concat_images_np
from ..utils.poco_utils import POCOUtils
from ..utils.save_results import SaveResults
from ..utils.renderer import Renderer
from ..utils.train_utils import is_main_process, get_index, freeze_model, decode_freeze_params
from ..utils.eval_utils import get_jnts_from_mesh, mpjpe_error, pampjpe_error, vert_error, calculate_pearson_coff
from ..dataset import BaseDataset, SMPLDataset, MixedDataset, EFTMixedDataset, TestAugDataset
from ..utils.geometry import estimate_translation, perspective_projection, rotation_matrix_to_angle_axis
from ..utils.image_utils import generate_part_labels, get_body_part_texture, get_default_camera

class LitModule(pl.LightningModule):

    def __init__(self, hparams):
        super(LitModule, self).__init__()

        self.save_hyperparameters(hparams)
        self.parse_hparams()
        self.new_lr = 0.

        self.model = POCO(
            backbone=self.hparams.POCO.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            uncert_layer=self.hparams.POCO.UNCERT_LAYER,
            activation_type=self.hparams.POCO.ACTIVATION_TYPE,
            uncert_type=self.hparams.POCO.UNCERT_TYPE,
            uncert_inp_type=self.hparams.POCO.UNCERT_INP_TYPE,
            loss_ver=self.hparams.POCO.LOSS_VER,
            num_neurons=self.hparams.POCO.NUM_NEURONS,
            num_flow_layers=self.hparams.POCO.NUM_FLOW_LAYERS,
            sigma_dim=self.hparams.POCO.SIGMA_DIM,
            num_nf_rv=self.hparams.POCO.NUM_NF_RV,
            mask_params_id=self.hparams.POCO.MASK_PARAMS_ID,
            nflow_mask_type=self.hparams.POCO.NFLOW_MASK_TYPE,
            exclude_uncert_idx=self.hparams.POCO.EXCLUDE_UNCERT_IDX,
            use_dropout=self.hparams.POCO.USE_DROPOUT,
            use_iter_feats=self.hparams.POCO.USE_ITER_FEATS,
            cond_nflow=self.hparams.POCO.COND_NFLOW,
            context_dim=self.hparams.POCO.CONTEXT_DIM,
            gt_pose_cond=self.hparams.POCO.GT_POSE_COND,
            gt_pose_cond_ds=self.hparams.POCO.GT_POSE_COND_DS,
            gt_pose_cond_ratio=self.hparams.POCO.GT_POSE_COND_RATIO,
            pretrained=self.hparams.TRAINING.PRETRAINED,
            is_test=self.hparams.RUN_TEST,
        )
        self.loss_fn = POCOLoss(
            shape_loss_weight=self.hparams.POCO.SHAPE_LOSS_WEIGHT,
            keypoint3d_loss_weight=self.hparams.POCO.KEYPOINT_3D_LOSS_WEIGHT,
            keypoint2d_loss_weight=self.hparams.POCO.KEYPOINT_2D_LOSS_WEIGHT,
            keypoint2d_noncrop=self.hparams.POCO.KEYPOINT_2D_NONCROP,
            pose_loss_weight=self.hparams.POCO.POSE_LOSS_WEIGHT,
            beta_loss_weight=self.hparams.POCO.BETA_LOSS_WEIGHT,
            openpose_train_weight=self.hparams.POCO.OPENPOSE_TRAIN_WEIGHT,
            gt_train_weight=self.hparams.POCO.GT_TRAIN_WEIGHT,
            pose_uncert_weight=self.hparams.POCO.POSE_UNCERT_WEIGHT,
            beta_uncert_weight=self.hparams.POCO.BETA_UNCERT_WEIGHT,
            jnt_uncert_weight=self.hparams.POCO.JNT_UNCERT_WEIGHT,
            nf_loss_weight=self.hparams.POCO.NF_LOSS_WEIGHT,
            genG_loss_weight=self.hparams.POCO.GENG_LOSS_WEIGHT,
            use_keyconf=self.hparams.POCO.USE_KEYCONF,
            loss_weight=self.hparams.POCO.LOSS_WEIGHT,
            loss_ver=self.hparams.POCO.LOSS_VER,
            uncert_type=self.hparams.POCO.UNCERT_TYPE,
            exclude_uncert_idx=self.hparams.POCO.EXCLUDE_UNCERT_IDX,
            use_smpl_render_loss=self.hparams.TRAINING.USE_SMPL_RENDER_LOSS,
            use_smpl_segm_loss=self.hparams.TRAINING.USE_SMPL_SEGM_LOSS,
            smpl_render_loss_weight=self.hparams.POCO.SMPL_RENDER_LOSS_WEIGHT,
            smpl_segm_loss_weight=self.hparams.POCO.SMPL_SEGM_LOSS_WEIGHT,
        )
        self.poco_utils = POCOUtils(self.hparams)

        self.smpl = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )

        self.smpl_native = SMPL_native(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.smpl.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
            uncert_type=self.hparams.POCO.UNCERT_TYPE,
        )

        if self.hparams.TRAINING.USE_SMPL_RENDER_LOSS or self.hparams.TRAINING.USE_SMPL_SEGM_LOSS:
            self.register_buffer(
                'body_part_texture',
                get_body_part_texture(self.smpl.faces, n_vertices=6890)
            )

            K, R, dist_coeffs = get_default_camera(focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                                                   img_size=self.hparams.DATASET.IMG_RES)

            self.register_buffer('K', K)
            self.register_buffer('R', R)
            self.register_buffer('dist_coeffs', dist_coeffs)
            self.register_buffer('smpl_faces', torch.from_numpy(self.smpl.faces.astype(np.int32)).unsqueeze(0))
            # bins are discrete part labels, add 1 to avoid quantization error
            n_parts = 24
            self.register_buffer('part_label_bins', (torch.arange(int(n_parts)) / float(n_parts) * 255.) + 1)

            self.neural_renderer = nr.Renderer(
                orig_size=self.hparams.DATASET.IMG_RES,
                image_size=self.hparams.DATASET.IMG_RES,
                light_intensity_ambient=1,
                light_intensity_directional=0,
                anti_aliasing=False,
            )


        self.train_ds = self.train_dataset()
        self.val_ds = self.validation_dataset()

        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )

        # Initialiatize variables related to evaluation
        self.best_result = math.inf
        self.best_pampjpe = math.inf
        self.best_mpjpe = math.inf
        self.best_v2v = math.inf
        self.best_mpjpe_var = math.inf
        self.best_corr = 0.
        self.best_epoch = 0
        self.val_accuracy_results = []

        # This class is used to save the predictions for later analysis
        self.save_results = SaveResults(self.hparams)

        # Initialize the validation variables
        if is_main_process():
            self.init_val_variables()


        self.pl_logging = self.hparams.PL_LOGGING
        self.pref_logger = self.hparams.PREF_LOGGER
        self.tb_idx = get_index(self.pref_logger, 'tensorboard')
        self.cmt_idx = get_index(self.pref_logger, 'comet')

        self.freeze_params_per_epoch = decode_freeze_params(
                                        self.hparams.TRAINING.FREEZE_PARAMS,
                                        self.hparams.TRAINING.MAX_EPOCHS)


    def parse_hparams(self):
        self.hparams.POCO.UNCERT_TYPE = list(filter(None, self.hparams.POCO.UNCERT_TYPE.split('-')))
        self.hparams.PREF_LOGGER = list(filter(None, self.hparams.PREF_LOGGER.split('-')))

    def init_val_variables(self):
        # stores mean mpjpe/pa-mpjpe values for all validation dataset samples
        self.val_mpjpe = [] # np.zeros(len(self.val_ds))
        self.val_pampjpe = [] # np.zeros(len(self.val_ds))
        self.val_v2v = []
        self.val_var = []

        self.save_results.init_variables()

    def forward(self):
        return None

    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)/(1024*1024)

    def total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())/(1024*1024)

    def on_train_epoch_start(self):

        if self.freeze_params_per_epoch is not None:
            fr_modules = self.freeze_params_per_epoch[self.current_epoch]
            # Unfreeze all modules
            freeze_model(self.model, freeze=False)
            # Freeze modules specified in the config
            for fr_mod in fr_modules:
                logger.warning(f'Freezing {fr_mod} in epoch - {self.current_epoch}')
                freeze_model(eval(f'self.model.{fr_mod}'), freeze=True)

            optim_conf = self.configure_optimizers()
            self.trainer.optimizers = [optim_conf['optimizer']]
            self.trainer.lr_schedulers = self.trainer._configure_schedulers(
                                            [optim_conf['lr_scheduler']],
                                            monitor="val_loss",
                                            is_manual_optimization=False)

    def training_step(self, batch, batch_nb):

        gt_keypoints_2d = batch['keypoints']  # 2D keypoints
        gt_pose = batch['pose']  # SMPL pose parameters
        gt_betas = batch['betas']  # SMPL beta parameters

        batch_size = gt_keypoints_2d.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(
            betas=gt_betas.contiguous(),
            body_pose=gt_pose[:, 3:].contiguous(),
            global_orient=gt_pose[:, :3].contiguous()
        )
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        if 'pose_3d' not in batch.keys():
            batch['pose_3d'] = gt_model_joints

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = \
            0.5 * self.hparams.DATASET.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_size=self.hparams.DATASET.IMG_RES,
        )

        batch['gt_cam_t'] = gt_cam_t
        batch['vertices'] = gt_vertices

        # Forward pass
        pred = self.model(batch)

        if self.hparams.TRAINING.USE_SMPL_RENDER_LOSS or self.hparams.TRAINING.USE_SMPL_SEGM_LOSS:
            batch['gt_segm_mask'], batch['gt_smpl_render'] = generate_part_labels(
                vertices=gt_vertices,
                faces=self.smpl_faces,
                cam_t=gt_cam_t,
                K=self.K,
                R=self.R,
                dist_coeffs=self.dist_coeffs,
                body_part_texture=self.body_part_texture,
                part_bins=self.part_label_bins,
                neural_renderer=self.neural_renderer,
            )

        if self.hparams.TRAINING.USE_SMPL_RENDER_LOSS or self.hparams.TRAINING.USE_SMPL_SEGM_LOSS:
            _, pred['pred_smpl_render'] = generate_part_labels(
                vertices=pred['smpl_vertices'],
                faces=self.smpl_faces,
                cam_t=pred['pred_cam_t'],
                K=self.K,
                R=self.R,
                dist_coeffs=self.dist_coeffs,
                body_part_texture=self.body_part_texture,
                part_bins=self.part_label_bins,
                neural_renderer=self.neural_renderer,
            )

        # Calculate the loss
        loss, loss_dict = self.loss_fn(pred=pred, gt=batch, cur_iteration=self.global_step)

        # Display functions for training samples
        if batch_nb % self.hparams.TRAINING.LOG_FREQ_TB_IMAGES == 0 \
                and is_main_process() and self.hparams.TRAINING.SAVE_IMAGES:
            self.train_summaries(input_batch=batch, output=pred)

        # Error Analysis
        if self.hparams.METHOD == 'poco' and is_main_process():
            self.poco_utils.accumulate_uncert('tr', pred, batch, batch_nb)
            if batch_nb % self.hparams.POCO.LOG_TRAIN_UNCERT == 0 and self.hparams.PL_LOGGING:
                self.poco_utils.log_uncert('tr', self.logger, self.global_step)

        # Sending metrics for logging
        for k, v in loss_dict.items():
            self.log(k, v, batch_size=batch_size, on_step=True, logger=True, sync_dist=True)

        return {'loss': loss}


    def validation_step(self, batch, batch_nb, vis=False):

        with torch.no_grad():
            pred = self.model(batch)
            pred_vertices = pred['smpl_vertices']
            pred_pose = pred['pred_pose']

        # For 3DPW get the 14 common joints from the rendered shape
        gt_keypoints_3d = batch['pose_3d']

        # Get 14 predicted joints from the mesh
        pred_keypoints_3d, pred_keypoints_3d_nonrel = \
                get_jnts_from_mesh(pred_vertices, self.J_regressor, self.val_ds.dataset)
        pred['pred_keypoints_3d_nonrel'] = pred_keypoints_3d_nonrel
        pred['pred_keypoints_3d'] = pred_keypoints_3d

        # Absolute error (MPJPE)
        error, error_per_joint = mpjpe_error(pred_keypoints_3d, gt_keypoints_3d)
        self.val_mpjpe += error.tolist()

        # Reconstuction_error
        r_error, r_error_per_joint = pampjpe_error(pred_keypoints_3d, gt_keypoints_3d, reduction=None)
        self.val_pampjpe += r_error.tolist()

        # Per-vertex error
        gt_vertices = batch['vertices'] if 'vertices' in batch.keys() else None
        v2v = vert_error(pred_vertices, gt_vertices)
        self.val_v2v += v2v.tolist()

        # Similarity between Confidence and OKS
        if self.hparams.METHOD == 'poco':
            uncert_type = self.hparams.POCO.UNCERT_TYPE[0]
            uncert = self.poco_utils.prepare_uncert(pred[f'var_{uncert_type}'], \
                                                  return_conf=False)
            pred['processed_uncert'] = uncert
            self.save_results.accumulate_corr_vect(pred, batch)

        self.save_results.accumulate_metrics(error_per_joint, r_error_per_joint, v2v)
        self.save_results.accumulate_preds(batch, pred)

        if vis:
            vis_image = self.validation_summaries(batch, pred, batch_nb, error, r_error)
            return {
                'vis_image': vis_image
                }

        if batch_nb % self.hparams.TESTING.LOG_FREQ_TB_IMAGES == 0 \
                and is_main_process() and self.hparams.TESTING.SAVE_IMAGES:
            self.validation_summaries(batch, pred, batch_nb, error, r_error)

        log_dict = {
            'val_loss': r_error.mean(),
            'val/val_mpjpe_step': error.mean(),
            'val/val_pampjpe_step': r_error.mean(),
            'val/val_v2v_step': v2v.mean(),
        }

        # Error Analysis
        if self.hparams.METHOD == 'poco' and is_main_process():
            val_var_step = self.poco_utils.accumulate_uncert('val', pred, batch, batch_nb)
            self.val_var += val_var_step.tolist()
        else:
            self.val_var += np.zeros_like(error).tolist()

        return log_dict


    def validation_epoch_end(self, val_step_outputs):

        batch_size = len(val_step_outputs)

        self.val_mpjpe = np.array(self.val_mpjpe)
        self.val_pampjpe = np.array(self.val_pampjpe)
        self.val_v2v = np.array(self.val_v2v)
        self.val_var = np.array(self.val_var)

        self.val_mpjpe_var = self.val_mpjpe / (self.val_var + 1e-9)
        avg_mpjpe, avg_pampjpe = 1000 * self.val_mpjpe.mean(), 1000 * self.val_pampjpe.mean()
        avg_v2v = 1000 * self.val_v2v.mean()
        avg_mpjpe_var = self.val_mpjpe_var.mean()
        avg_var = self.val_var.mean()

        avg_corr = 0.
        if self.hparams.METHOD == 'poco':
            corr_x, corr_y = self.save_results.get_corr_vect()
            avg_corr,_ = calculate_pearson_coff(corr_x, corr_y)

        logger.info(f'***** Epoch {self.current_epoch} *****')
        logger.info('MPJPE: ' + str(avg_mpjpe))
        logger.info('PA-MPJPE: ' + str(avg_pampjpe))
        logger.info('V2V (mm): ' + str(avg_v2v))
        logger.info('Var-MPJPE: ' + str(avg_mpjpe_var))
        logger.info('Variance: ' + str(avg_var))
        logger.info('Uncert Error Correlation: ' + str(avg_corr))

        avg_mpjpe, avg_pampjpe, avg_v2v = torch.tensor(avg_mpjpe), torch.tensor(avg_pampjpe), torch.tensor(avg_v2v)
        avg_var, avg_corr = torch.tensor(avg_var), torch.tensor(avg_corr)


        acc = {
            'val_mpjpe': avg_mpjpe.item(),
            'val_pampjpe': avg_pampjpe.item(),
            'val_v2v': avg_v2v.item(),
        }
        self.val_save_best_results(acc)

        if self.hparams.METHOD == 'poco' and self.hparams.PL_LOGGING and is_main_process():
            self.poco_utils.log_uncert('val', self.logger, self.current_epoch)

        # Best model selection criterion - 1.5 * PAMPJPE + MPJPE
        best_result = 0.5 * (1.5 * avg_pampjpe.clone().cpu().numpy() + avg_mpjpe.clone().cpu().numpy())
        if best_result < self.best_result:
            logger.info(f'best model found: current-> {best_result} | previous-> {self.best_result}')
            self.best_result = best_result
            self.best_pampjpe = avg_pampjpe
            self.best_mpjpe = avg_mpjpe
            self.best_v2v = avg_v2v
            self.best_var = avg_var
            self.best_corr = avg_corr
            self.best_epoch = self.current_epoch

            if is_main_process():
                self.save_results.dump_results(self.current_epoch)
                torch.save(self.model.state_dict(),
                           f'{self.hparams.LOG_DIR}/best_model.pt')

        # If pose metric are same, save the model with better error uncertainty corr
        elif best_result == self.best_result and \
             self.hparams.METHOD == 'poco' and \
             avg_corr > self.best_corr:

            logger.info(f'Pose metrics are same, saving model with better uncert_error corr')
            logger.info(f'current_corr-> {avg_corr.item()} | previous_corr-> {self.best_corr.item()}')
            self.best_result = best_result
            self.best_pampjpe = avg_pampjpe
            self.best_mpjpe = avg_mpjpe
            self.best_v2v = avg_v2v
            self.best_var = avg_var
            self.best_corr = avg_corr
            self.best_epoch = self.current_epoch

            if is_main_process():
                self.save_results.dump_results(self.current_epoch)
                torch.save(self.model.state_dict(),
                           f'{self.hparams.LOG_DIR}/best_model.pt')

        log_dict = {
            'val_loss': best_result,
            'val/val_mpjpe': avg_mpjpe,
            'val/val_pampjpe': avg_pampjpe,
            'val/val_v2v': avg_v2v,
            'val/val_var': avg_var,
            'val/val_mpjpe_var': avg_mpjpe_var,
            'val/val_corr': avg_corr,
            'val/best_pampjpe': self.best_pampjpe,
            'val/best_mpjpe': self.best_mpjpe,
            'val/best_v2v': self.best_v2v,
            'val/best_var': self.best_var,
            'val/best_mpjpe_var': self.best_mpjpe_var,
            'val/best_corr': self.best_corr,
            'val/best_epoch': self.best_epoch,
            'step': self.current_epoch,
        }

        for k, v in log_dict.items():
            self.log(k, v, batch_size=batch_size, on_epoch=True, logger=True, sync_dist=True)

        self.init_val_variables()

    def train_summaries(self, input_batch, output):

        if self.pl_logging == True:
            tb_logger = self.logger[self.tb_idx] if self.tb_idx is not None else None
            comet_logger = self.logger[self.cmt_idx] if self.cmt_idx is not None else None

        images = input_batch['img']

        pred_vertices = output['smpl_vertices'].detach()
        gt_vertices = input_batch['vertices']

        pred_cam_t = output['pred_cam_t'].detach()
        gt_cam_t = input_batch['gt_cam_t']

        pred_kp_2d = output['pred_kp2d'].detach() if 'pred_kp2d' in output.keys() else None

        grphs = input_batch['grph'] if 'grph' in input_batch.keys() else None

        var = None
        uncert_type = self.hparams.POCO.UNCERT_TYPE[0] # Render with first uncertainty type
        if f'var_{uncert_type}' in output.keys():
            var = output[f'var_{uncert_type}']
            var = self.poco_utils.prepare_uncert(var)

        pred_pose = output['pred_pose'].detach()
        pred_shape = output['pred_shape'].detach()

        images_gt = self.renderer.visualize_tb(
            vertices=gt_vertices,
            camera_translation=gt_cam_t,
            images=images,
            # kp_2d=input_batch['keypoints'],
            sideview=self.hparams.TESTING.SIDEVIEW,
        )
        images_pred = self.renderer.visualize_tb(
            vertices=pred_vertices,
            camera_translation=pred_cam_t,
            images=images,
            pose=pred_pose,
            betas=pred_shape,
            var=var,
            var_dtype='tr',
            sideview=self.hparams.TESTING.SIDEVIEW,
        )

        if self.pl_logging == True:
            tb_logger.experiment.add_image('rend_tr_pred_shape', images_pred, self.global_step)
            tb_logger.experiment.add_image('rend_tr_gt_shape', images_gt, self.global_step)

        if self.hparams.TRAINING.SAVE_IMAGES == True:
            images_gt = images_gt.cpu().numpy().transpose(1, 2, 0) * 255
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255

            images_gt = np.clip(images_gt, 0, 255).astype(np.uint8)
            images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)

            images_save = concat_images_np(images_gt, images_pred)

            save_dir = os.path.join(self.hparams.LOG_DIR, 'train_output_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                    os.path.join(save_dir, f'result_{self.global_step:05d}.png'),
                cv2.cvtColor(images_save, cv2.COLOR_BGR2RGB)
            )

    def validation_summaries(self, input_batch, output, batch_idx, error=None, r_error=None):

        if self.pl_logging:
            tb_logger = self.logger[self.tb_idx] if self.tb_idx is not None else None
            comet_logger = self.logger[self.cmt_idx] if self.cmt_idx is not None else None

        images = input_batch['img']
        images_gt = None

        pred_vertices = output['smpl_vertices'].detach()
        pred_cam_t = output['pred_cam_t'].detach()
        pred_kp_2d = output['pred_kp2d'].detach() if 'pred_kp2d' in output.keys() else None

        var = None
        uncert_type = self.hparams.POCO.UNCERT_TYPE[0] # Render with first uncertainty type
        if f'var_{uncert_type}' in output.keys():
            var = output[f'var_{uncert_type}']
            var = self.poco_utils.prepare_uncert(var)

        pred_pose = output['pred_pose'].detach() if 'pred_pose' in output.keys() else None
        pred_shape = output['pred_shape'].detach() if 'pred_shape' in output.keys() else None

        images_pred = self.renderer.visualize_tb(
            vertices=pred_vertices,
            camera_translation=pred_cam_t,
            images=images,
            pose=pred_pose,
            betas=pred_shape,
            var=var,
            var_dtype='val',
            sideview=self.hparams.TESTING.SIDEVIEW,
            display_all=self.hparams.TESTING.DISP_ALL,
        )

        if self.pl_logging == True:
            tb_logger.experiment.add_image('rend_val_pred_shape', images_pred, self.global_step)

        if self.hparams.TESTING.SAVE_IMAGES == True:
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
            images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)

            images_save = images_pred

            save_dir = os.path.join(self.hparams.LOG_DIR, 'val_output_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                    os.path.join(save_dir, f'result_{self.global_step:05d}_{batch_idx}.png'),
                cv2.cvtColor(images_save, cv2.COLOR_BGR2RGB)
            )
        else:
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
            return np.clip(images_pred, 0, 255).astype(np.uint8)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        module_lr = self.hparams.OPTIMIZER.MODULE_LR.split('_')
        if len(module_lr) == 4:
            module_lr = [float(x) for x in module_lr]
            global_lr = self.hparams.OPTIMIZER.LR
            optimizer = torch.optim.Adam(
                [
                    {"params": self.model.backbone.parameters(), "lr": module_lr[0] * global_lr},
                    {"params": self.model.head.parameters(), "lr": module_lr[1] * global_lr},
                    {"params": self.model.uncert_head.parameters(), "lr": module_lr[2] * global_lr},
                    {"params": self.model.flow_head.parameters(), "lr": module_lr[3] * global_lr},
                ],
                weight_decay=self.hparams.OPTIMIZER.WD,
                amsgrad=self.hparams.OPTIMIZER.AMSGRAD,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.OPTIMIZER.LR,
                weight_decay=self.hparams.OPTIMIZER.WD,
                amsgrad=self.hparams.OPTIMIZER.AMSGRAD,
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            cooldown=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },}


    def train_dataset(self):

        if self.hparams.DATASET.TRAIN_DS == 'all':
            train_ds = EFTMixedDataset(
                self.hparams.DATASET,
                self.hparams.METHOD,
                use_augmentation=self.hparams.TRAINING.USE_AUGM,
                ignore_3d=self.hparams.DATASET.IGNORE_3D,
                num_images=self.hparams.DATASET.NUM_IMAGES,
                is_train=True
            )
        elif self.hparams.DATASET.TRAIN_DS == 'stage':
            stage_datasets = self.hparams.DATASET.STAGE_DATASETS.split(',')
            stage_dicts = {x.split('+')[0]: ''.join(x.split('+')[1:]) for x in stage_datasets}

            if str(self.current_epoch) in stage_dicts.keys():
                self.hparams.DATASET.DATASETS_AND_RATIOS = stage_dicts[str(self.current_epoch)]

            train_ds = EFTMixedDataset(
                self.hparams.DATASET,
                self.hparams.METHOD,
                use_augmentation=self.hparams.TRAINING.USE_AUGM,
                ignore_3d=self.hparams.DATASET.IGNORE_3D,
                num_images=self.hparams.DATASET.NUM_IMAGES,
                is_train=True
            )

        elif self.hparams.DATASET.TRAIN_DS in config.DATASET_FOLDERS.keys():
            train_ds = eval(f'{self.hparams.DATASET.DATASET_TYPE}')(
                self.hparams.DATASET,
                self.hparams.METHOD,
                use_augmentation=self.hparams.TRAINING.USE_AUGM,
                dataset=self.hparams.DATASET.TRAIN_DS,
                num_images=self.hparams.DATASET.NUM_IMAGES,
            )
        else:
            logger.error(f'{self.hparams.DATASET.TRAIN_DS} is undefined!')
            exit()
        return train_ds

    def validation_dataset(self):
        if self.hparams.METHOD == 'poco':
            self.hparams.DATASET.TEST_ROT = self.hparams.TESTING.TEST_ROT
            self.hparams.DATASET.TEST_SCALE = self.hparams.TESTING.TEST_SCALE

        val_ds = eval(f'{self.hparams.TESTING.DATASET_TYPE}')(
            self.hparams.DATASET,
            self.hparams.METHOD,
            dataset=self.hparams.DATASET.VAL_DS,
            num_images=self.hparams.DATASET.NUM_IMAGES,
            is_train=False,
        )
        return val_ds

    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            shuffle=self.hparams.DATASET.SHUFFLE_VAL,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def val_save_best_results(self, acc):
        json_file = os.path.join(self.hparams.LOG_DIR, 'val_accuracy_results.json')
        self.val_accuracy_results.append([self.global_step, acc])
        with open(json_file, 'w') as f:
            json.dump(self.val_accuracy_results, f, indent=4)
