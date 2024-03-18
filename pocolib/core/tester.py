# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# Code Modified from: https://github.com/mkocabas/PARE/blob/master/pare/core/tester.py

import os
import cv2
import time
import torch
import joblib
import colorsys
import numpy as np
from os.path import isfile, isdir, join, basename
from tqdm import tqdm
from loguru import logger
from yolov3.yolo import YOLOv3
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from . import config
from ..models import POCO, SMPL, HMR
from .config import update_hparams
from ..utils.vibe_renderer import Renderer
from ..utils.pose_tracker import run_posetracker
from ..dataset.inference import Inference
from ..utils.smooth_pose import smooth_pose
from ..utils.poco_utils import POCOUtils
from ..utils.image_utils import overlay_text, calculate_bbox_info, calculate_focal_length
from ..utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    prepare_rendering_results,
    xyhw_to_xyxy,
)
from ..utils.vibe_image_utils import get_single_image_crop_demo


MIN_NUM_FRAMES = 0


class POCOTester:
    def __init__(self, args):
        self.args = args
        cfg_file = self.args.cfg
        self.model_cfg = update_hparams(cfg_file)
        self.model_cfg.POCO.KINEMATIC_UNCERT = self.args.no_kinematic_uncert
        self.ptfile = self.args.ckpt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.uncert_type = self.model_cfg.POCO.UNCERT_TYPE.split('-')[0] #render with 1st uncerttype
        self.loss_ver = self.model_cfg.POCO.LOSS_VER
        self.poco_utils = POCOUtils(self.model_cfg)
        self.model = self._build_model()
        self.model.eval()

        self.smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to('cuda')


    def _build_model(self):
        # ========= Define POCO model ========= #
        model_cfg = self.model_cfg
        if model_cfg.METHOD == 'poco':
            model = POCO(
                backbone=model_cfg.POCO.BACKBONE,
                img_res=model_cfg.DATASET.IMG_RES,
                uncert_layer=model_cfg.POCO.UNCERT_LAYER,
                activation_type=model_cfg.POCO.ACTIVATION_TYPE,
                uncert_type=model_cfg.POCO.UNCERT_TYPE,
                uncert_inp_type=model_cfg.POCO.UNCERT_INP_TYPE,
                loss_ver=model_cfg.POCO.LOSS_VER,
                num_neurons=model_cfg.POCO.NUM_NEURONS,
                num_flow_layers=model_cfg.POCO.NUM_FLOW_LAYERS,
                sigma_dim=model_cfg.POCO.SIGMA_DIM,
                num_nf_rv=model_cfg.POCO.NUM_NF_RV,
                mask_params_id=model_cfg.POCO.MASK_PARAMS_ID,
                nflow_mask_type=model_cfg.POCO.NFLOW_MASK_TYPE,
                exclude_uncert_idx=model_cfg.POCO.EXCLUDE_UNCERT_IDX,
                use_dropout=model_cfg.POCO.USE_DROPOUT,
                use_iter_feats=model_cfg.POCO.USE_ITER_FEATS,
                cond_nflow=model_cfg.POCO.COND_NFLOW,
                context_dim=model_cfg.POCO.CONTEXT_DIM,
                gt_pose_cond=model_cfg.POCO.GT_POSE_COND,
                gt_pose_cond_ratio=model_cfg.POCO.GT_POSE_COND_RATIO,
                pretrained=self.ptfile,
                inf_model=self.args.inf_model,
            ).to(self.device)
            self.backbone = model_cfg.POCO.BACKBONE
        elif model_cfg.METHOD == 'spin':
            model = HMR(
                backbone=model_cfg.SPIN.BACKBONE,
                img_res=model_cfg.DATASET.IMG_RES,
                pretrained=self.ptfile,
            ).to(self.device)
            self.backbone = model_cfg.SPIN.BACKBONE
        else:
            logger.error(f'{model_cfg.METHOD} is undefined!')
            exit()

        return model

    def run_tracking(self, video_file, image_folder, output_folder):
        # ========= Run tracking ========= #
        if self.args.tracking_method == 'bbox':
            # run multi object tracker
            mot = MPT(
                device=self.device,
                batch_size=self.args.tracker_batch_size,
                display=self.args.display,
                detector_type=self.args.detector,
                output_format='dict',
                yolo_img_size=self.args.yolo_img_size,
            )
            tracking_results = mot(image_folder)
        elif self.args.tracking_method == 'pose':
            if not os.path.isabs(video_file):
                video_file = os.path.join(os.getcwd(), video_file)
            tracking_results = run_posetracker(video_file, staf_folder=self.args.staf_dir, display=self.args.display)
        else:
            logger.error(f'Tracking method {self.args.tracking_method} is not defined')

        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]

        return tracking_results

    def run_detector(self, image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=self.args.display,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = mot.detect(image_folder)
        return bboxes

    @torch.no_grad()
    def run_on_image_folder(self, image_folder, detections, output_path, output_img_folder, bbox_scale=1.0):
        image_file_names = [
            join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        ]
        image_file_names = sorted(image_file_names)


        imgnames_, scales_, centers_, Ss_, parts_, openposes_, poses_, shapes_, vars_ = \
                [], [], [], [], [], [], [], [], []

        pred_cam_, orig_cam_, verts_, betas_, pose_, joints3d_, smpl_joints2d_, bboxes_, var_ = [], [], [], [], [], [], [], [], []
        for img_idx in range(0, len(image_file_names), self.args.skip_frame):
            img_fname = image_file_names[img_idx]
            dets = detections[img_idx]

            img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            orig_height, orig_width = img.shape[:2]

            if len(dets) < 1:
                logger.warning(f'No YOLO detections found for image - {img_fname}')
                continue

            inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
                                     self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

            bbox_info, focal_lengths, orig_shapes, scales, centers = [], [], [], [], []
            for det_idx, det in enumerate(dets):
                bbox = det
                norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                    img,
                    bbox,
                    kp_2d=None,
                    scale=bbox_scale,
                    crop_size=self.model_cfg.DATASET.IMG_RES
                )
                inp_images[det_idx] = norm_img.float().to(self.device)

                #TODO: Optimise
                center = [bbox[0], bbox[1]]
                orig_shape = [orig_height, orig_width]
                scale = max(bbox[2], bbox[3])/200.

                centers.append(center)
                orig_shapes.append(orig_shape)
                scales.append(scale)

                bbox_info.append(calculate_bbox_info(center, scale, orig_shape))
                focal_lengths.append(calculate_focal_length(orig_height, orig_width))

            batch = {
                     'img': inp_images,
                     'bbox_info': torch.FloatTensor(bbox_info).cuda(),
                     'focal_length': torch.FloatTensor(focal_lengths).cuda(),
                     'scale': torch.FloatTensor(scales).cuda(),
                     'center': torch.FloatTensor(centers).cuda(),
                     'orig_shape': torch.FloatTensor(orig_shapes).cuda(),
                    }
            output = self.model(batch)

            pred_cam = output['pred_cam'].cpu().numpy()
            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=dets,
                img_width=orig_width,
                img_height=orig_height
            )
            smpl_joints2d = output['smpl_joints2d'].cpu().numpy()

            # In CLIFF, the keypoints are in original image space
            if not 'cliff' in self.backbone:
                smpl_joints2d = convert_crop_coords_to_orig_img(
                    bbox=dets,
                    keypoints=smpl_joints2d,
                    crop_size=self.model_cfg.DATASET.IMG_RES,
                )

            smpl_joints2d = np.concatenate([smpl_joints2d, \
                    np.ones((len(dets), 49, 1))], axis=-1)
            
            output['bboxes'] = dets
            output['orig_cam'] = orig_cam
            output['crop_img'] = raw_img
            output['crop_cam'] = output['pred_cam']
            output['smpl_joints2d'] = smpl_joints2d

            variance = None
            if f'var_{self.uncert_type}' in output.keys():
                variance = self.poco_utils.prepare_uncert(output[f'var_{self.uncert_type}'])
                variance_global = self.poco_utils.get_global_uncert(variance.copy())
                variance_global = np.clip(variance_global, 0, 0.99)

            del inp_images

            if not self.args.no_render:

                for k,v in output.items():
                    if output[k] is not None:
                        if isinstance(v, torch.Tensor):
                            output[k] = v.cpu().numpy()

                if self.args.render_crop == True:
                    img = output['crop_img']
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    orig_img = img.copy()
                    res = [224,224]
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    orig_img = img.copy()
                    res = [orig_width, orig_height]

                renderer = Renderer(
                    resolution=res,
                    orig_img=True,
                    wireframe=self.args.wireframe,
                    uncert_type=self.uncert_type,
                )

                if self.args.sideview:
                    side_img = np.ones_like(img) * 255

                for idx in range(len(dets)):
                    if idx > 0 and self.args.render_crop:
                        continue
                    verts = output['smpl_vertices'][idx]
                    if self.args.render_crop == True:
                        cam = output['crop_cam'][idx]
                        cam = [cam[0], cam[0], cam[1], cam[2]]
                    else:
                        cam = output['orig_cam'][idx]
                    keypoints = output['smpl_joints2d'][idx]


                    if self.args.no_uncert_color:
                        mc = [0.70, 0.70, 0.70]
                        rend_var, var_global = None, -1
                    else:
                        var = variance[idx]
                        var_global = variance_global[idx]
                        mc = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
                        rend_var = var.copy()


                    mesh_filename = None

                    if self.args.save_obj:
                        mesh_folder = join(output_path, 'meshes', basename(img_fname).split('.')[0])
                        os.makedirs(mesh_folder, exist_ok=True)
                        mesh_filename = join(mesh_folder, f'{idx:06d}.obj')

                    print(f'inside tester-> {rend_var}')
                    img = renderer.render(
                        img,
                        verts,
                        cam=cam,
                        var=rend_var.copy(),
                        color=mc,
                        mesh_filename=mesh_filename,
                        hps_backbone=self.backbone,
                    )

                    ### Overlay text for debug
                    # overlay_str = f'frame:{img_idx:05d} person:{idx:02d} var:{var_global:.3f}'
                    # log_str = f'img_f:{img_fname} person:{idx:02d} var:{var_global:.3f}'
                    # if not (self.args.render_crop or self.args.no_uncert_color):
                    #     img = overlay_text(img, overlay_str, idx+1)
                    # with open(f'{output_path}/uncertainty.log', "a") as f:
                    #     print(log_str, file=f)

                    if self.args.draw_keypoints:
                        for _, pt in enumerate(keypoints[25:]): # SMPL Keypoints
                            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1)
                        for _, pt in enumerate(keypoints[:25]): # Openpose keypoints if available
                            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 0), -1)

                    ### Overlay text for debug
                    # bbox_xyxy = xyhw_to_xyxy(output['bboxes'][idx].copy())
                    # x_min, y_min, x_max, y_max = bbox_xyxy
                    # cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 3)

                    if self.args.sideview:
                        side_img = renderer.render(
                            side_img,
                            verts,
                            cam=cam,
                            var=rend_var,
                            color=mc,
                            angle=270,
                            axis=[0, 1, 0],
                            hps_backbone=self.backbone,
                        )

                if self.args.sideview:
                    img = np.concatenate([img, side_img], axis=1)

                cv2.imwrite(join(output_img_folder, basename(img_fname)), img)

                if self.args.display:
                    cv2.imshow('Video', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            logger.info(f'Processed image - {img_idx}')

        if self.args.display:
            cv2.destroyAllWindows()

    @torch.no_grad()
    def run_on_video(self, tracking_results, image_folder, orig_width, orig_height, bbox_scale=1.0):
        # ========= Run poco on each person ========= #
        logger.info(f'Running poco on each tracklet...')

        poco_results = {}
        for person_id in tqdm(list(tracking_results.keys())):
            bboxes = joints2d = frame_kps = None

            if self.args.tracking_method == 'bbox':
                bboxes = tracking_results[person_id]['bbox']
            elif self.args.tracking_method in ['pose']:
                joints2d = tracking_results[person_id]['joints2d']

            frames = tracking_results[person_id]['frames']

            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                frame_kps=frame_kps,
                scale=bbox_scale,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            frame_kps = dataset.frame_kps

            has_keypoints = True if joints2d is not None else False
            has_frame_kps = True if frame_kps is not None else False

            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=8)

            pred_cam, pred_cam_t, pred_verts, pred_verts_opt, pred_cam_opt, pred_pose, pred_betas, pred_var, pred_var_global, \
            pred_joints3d, pred_joints2d, joints2d_op, joints2d_op_orig = [], [], [], [], [], [], [], [], [], [], [], [], []

            for batch_dict in dataloader:
                if torch.all(batch_dict['has_frame_kps']):
                    j2d, j2d_orig = batch_dict['j2d'], batch_dict['j2d_orig']
                    joints2d_op.append(j2d.reshape(-1, 44, 3).to(self.device))
                    joints2d_op_orig.append(j2d_orig.reshape(-1, 44, 3))

                batch_dict = {k: v.to(device=self.device, non_blocking=True) \
                              if hasattr(v, 'to') else v for k, v in batch_dict.items()}

                output = self.model(batch_dict)

                pred_cam.append(output['pred_cam'])  # [:, :, :3].reshape(batch_size, -1))
                pred_cam_t.append(output['pred_cam_t'])  # [:, :, :3].reshape(batch_size, -1))
                pred_verts.append(output['smpl_vertices'])  # .reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['pred_pose'])  # [:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['pred_shape'])  # [:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['smpl_joints3d'])  # .reshape(batch_size * seqlen, -1, 3))
                pred_joints2d.append(output['smpl_joints2d'])
                if f'var_{self.uncert_type}' in output.keys():
                    var = self.poco_utils.prepare_uncert(output[f'var_{self.uncert_type}'], True)
                    pred_var.append(var.clone())
                    var_global = self.poco_utils.get_global_uncert(var.clone())
                    pred_var_global.append(var_global)

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_cam_t = torch.cat(pred_cam_t, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            pred_joints2d = torch.cat(pred_joints2d, dim=0)
            if f'var_{self.uncert_type}' in output.keys():
                pred_var = torch.cat(pred_var, dim=0).cpu().numpy()
                pred_var_global = torch.cat(pred_var_global, dim=0).cpu().numpy()

            del batch_dict

            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            pred_joints2d = pred_joints2d.cpu().numpy()

            if self.args.smooth:
                min_cutoff = self.args.min_cutoff  # 0.004
                beta = self.args.beta  # 1.5
                logger.info(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
                pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                                   min_cutoff=min_cutoff, beta=beta)

            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )
            logger.info('Converting smpl keypoints 2d to original image coordinate')

            pred_joints2d = convert_crop_coords_to_orig_img(
                bbox=bboxes,
                keypoints=pred_joints2d,
                crop_size=self.model_cfg.DATASET.IMG_RES,
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints2d': joints2d,
                'smpl_joints3d': pred_joints3d,
                'smpl_joints2d': pred_joints2d,
                'var': pred_var,
                'var_global': pred_var_global,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            poco_results[person_id] = output_dict
        return poco_results

    def render_results(self, poco_results, image_folder, output_img_folder, output_path,
                       orig_width, orig_height, num_frames):
        # ========= Render results as a single video ========= #
        renderer = Renderer(
            resolution=(orig_width, orig_height),
            orig_img=True,
            wireframe=self.args.wireframe,
            uncert_type=self.uncert_type,
        )

        logger.info(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(poco_results, num_frames)
        # mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in poco_results.keys()}
        mesh_color = {k: [0.98, 0.54, 0.44] for k in poco_results.keys()}

        image_file_names = sorted([
            join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)
            orig_img = img.copy()

            if self.args.sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']
                frame_kp = person_data['joints2d']
                frame_var = person_data['var']
                frame_var_global = person_data['var_global']
                if frame_var_global:
                    frame_var_global = np.clip(frame_var_global, 0, 0.99)

                mc = mesh_color[person_id]

                if frame_var is None:
                    rend_var, frame_var_global = None, -1
                    mc = [0.70, 0.70, 0.70]
                else:
                    rend_var = frame_var.copy()

                mesh_filename = None

                if self.args.save_obj:
                    mesh_folder = join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = join(mesh_folder, f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    var=rend_var,
                    color=mc,
                    mesh_filename=mesh_filename,
                    hps_backbone=self.backbone,
                )
                overlay_str = f'frame:{frame_idx:05d} person:{person_id:02d} var:{frame_var_global:.3f}'
                log_str = f'img_f:{img_fname} person:{person_id:02d} var:{frame_var_global:.3f}'
                # img = overlay_text(img, overlay_str, person_id)
                with open(f'{output_path}/uncertainty.log', "a") as f:
                    print(log_str, file=f)

                if self.args.draw_keypoints:
                    for idx, pt in enumerate(frame_kp):
                        cv2.circle(img, (pt[0], pt[1]), 4, (0,255,0), -1)

                if self.args.sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        var=rend_var,
                        color=mc,
                        angle=270,
                        axis=[0, 1, 0],
                        hps_backbone=self.backbone,
                    )
                    side_img = overlay_text(side_img, f'Other View')

            if self.args.sideview:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if self.args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.args.display:
            cv2.destroyAllWindows()
