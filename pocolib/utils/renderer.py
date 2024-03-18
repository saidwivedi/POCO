import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] \
#     if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

import io
import cv2
import re
import time
import torch
import joblib
import trimesh
import pyrender
import numpy as np
from loguru import logger
from os.path import isfile
from torchvision.utils import make_grid
from ..core import constants

import matplotlib.pyplot as plt
from matplotlib import cm as col_map, colors
from ..utils.image_utils import show_imgs, overlay_text
from ..utils.poco_utils import get_kinematic_uncert

from . import kp_utils
from ..core.config import SMPL_MODEL_DIR

def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
        'pinkish': np.array([204, 77, 77]),
    }
    return colors

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None, mesh_color='light_pink', uncert_type=[]):
        self.img_res = img_res
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.mesh_color = get_colors()[mesh_color]
        self.uncert_type = uncert_type # use the first uncertainty type for rendering

    def de_norm(self, images):
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        return images

    def visualize_tb(self, vertices, camera_translation, images, grphs=None, var=None, var_dtype=None, \
                     pose=None, betas=None, kp_2d=None, joint_labels=None, \
                     nb_max_img=6, sideview=False, display_all=True, hps_backbone='cliff'):

        images = self.de_norm(images)
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))

        if torch.is_tensor(grphs):
            grphs = grphs.cpu()
        if not isinstance(var, np.ndarray):
            var = [None] * vertices.shape[0]
        if torch.is_tensor(pose):
            pose = pose
        if torch.is_tensor(betas):
            betas = betas
        if kp_2d is not None:
            kp_2d = kp_2d.cpu().numpy()
        self.var_dtype = []
        if var_dtype is not None:
            self.var_dtype.append(var_dtype)

        rend_imgs = []
        nb_max_img = min(nb_max_img, vertices.shape[0])
        for i in range(nb_max_img):

            rend_img = torch.from_numpy(
                np.transpose(self.render(vertices[i],
                             camera_translation[i].copy(),
                             images_np[i], var[i], hps_backbone=hps_backbone),
                (2,0,1))
            ).float()

            if display_all:
                rend_imgs.append(images[i])
                if kp_2d is not None:
                    kp_img = draw_skeleton(images_np[i].copy(), kp_2d=kp_2d[i], dataset='openpose')
                    kp_img = torch.from_numpy(np.transpose(kp_img, (2,0,1))).float()
                    rend_imgs.append(kp_img)

                if torch.is_tensor(grphs):
                    rend_imgs.append(grphs[i])

                uncert_imgs = []
            rend_imgs.append(rend_img)

            if sideview:
                side_img = torch.from_numpy(
                    np.transpose(
                        self.render(vertices[i], camera_translation[i], np.ones_like(images_np[i]), var[i], sideview=True, hps_backbone=hps_backbone),
                        (2,0,1)
                    )
                ).float()
                rend_imgs.append(side_img)

        nrow = 1
        if sideview: nrow += 1
        if display_all:
            nrow += 1
            nrow += len(uncert_imgs)
            if kp_2d is not None: nrow += 1
            if joint_labels is not None: nrow += 1
            if torch.is_tensor(grphs): nrow += 1

        rend_imgs = make_grid(rend_imgs, nrow=nrow)
        return rend_imgs

    def render(self, vertices, camera_translation, image, var=None, sideview=False, hps_backbone='cliff'):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(self.mesh_color[0] / 255., self.mesh_color[1] / 255., self.mesh_color[2] / 255., 1.0))

        camera_translation[0] *= -1.

        if isinstance(var, np.ndarray) and len(self.uncert_type) > 0:
            vertex_colors = get_vertex_colors(var.copy(), hps_backbone)
            mesh = trimesh.Trimesh(vertices, self.faces, vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices, self.faces)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        if sideview:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(270), [0, 1, 0])
            mesh.apply_transform(rot)

        if isinstance(var, np.ndarray):
            mesh = pyrender.Mesh.from_trimesh(mesh)
        else:
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        return output_img

def get_vertex_colors(per_joint_label, hps_backbone='cliff', sensitivity_threshold=0.40):
    """
    color vertices based on a per_joint_label, joints are native SMPL joints
    per_joint_label np.array (24,)
    """

    smpl_segmentation = joblib.load('data/smpl_partSegmentation_mapping.pkl')
    n_vertices = smpl_segmentation['smpl_index'].shape[0]

    vertex_colors = np.ones((n_vertices, 4)) * np.array([0.3, 0.3, 0.3, 1])
    cm = col_map.get_cmap('jet')

    vmax, vmin = 1, 0

    if per_joint_label.shape[0] > 1:
        # for cliff, only taking the global uncertainty works better
        if 'cliff' in hps_backbone:
            if per_joint_label[0] > 2*sensitivity_threshold: # If hip uncert greater than threshold, wholebody uncert
                vmax = per_joint_label[0]
            per_joint_label[:] = per_joint_label[0]
        elif 'pare' in hps_backbone:
            if per_joint_label[0] > sensitivity_threshold: # If hip uncert greater than threshold, wholebody uncert
                vmax = per_joint_label[0]
            per_joint_label[:] = per_joint_label[:].mean()
    else:
        per_joint_label = np.repeat(per_joint_label, 24)

    for idx, label in enumerate(list(per_joint_label)):
        norm_gt = colors.Normalize(vmin=vmin, vmax=vmax)
        vertex_colors[smpl_segmentation['smpl_index'] == idx] = cm(norm_gt(label))

    return vertex_colors

def get_arr_from_figure(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr

def draw_skeleton(image, kp_2d, dataset='common', unnormalize=True, thickness=2):
    image = image * 255
    image = np.clip(image, 0, 255)

    if unnormalize:
        kp_2d[:,:2] = 0.5 * 224 * (kp_2d[:, :2] + 1) # normalize_2d_kp(kp_2d[:,:2], 224, inv=True)

    if kp_2d.shape[1] == 2:
        kp_2d = np.hstack([kp_2d, np.ones((kp_2d.shape[0], 1))])

    kp_2d[:,2] = kp_2d[:,2] > 0.3
    kp_2d = np.array(kp_2d, dtype=int)


    rcolor = [255,0,0]
    pcolor = [0,255,0]
    lcolor = [0,0,255]

    skeleton = eval(f'kp_utils.get_{dataset}_skeleton')()

    # common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
    for idx, pt in enumerate(kp_2d):
        if pt[2] > 0: # if visible
        # cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)
            cv2.putText(image, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

    # for i,(j1,j2) in enumerate(skeleton):
    #     # if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
    #     # if dataset == 'common':
    #     #     color = rcolor if common_lr[i] == 0 else lcolor
    #     # else:
    #     if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
    #         color = lcolor if i % 2 == 0 else rcolor
    #         pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])
    #         cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    image = np.asarray(image, dtype=float) / 255
    return image

def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0)/(2*ratio)

    return kp_2d


def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
        'confident': np.array([0, 111, 255]),
        'uncertain': np.array([255, 0, 0]),
    }
    return colors

def get_smpl_faces():
    from smplx import SMPL
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces
