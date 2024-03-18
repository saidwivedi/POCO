"""
This file contains functions that are used to perform data augmentation.
"""
import torch
import numpy as np
import scipy.misc
import cv2
import math
import joblib
from trimesh.visual import color

import jpeg4py as jpeg
from skimage.transform import rotate, resize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from ..core import constants
from .vibe_image_utils import gen_trans_from_patch_cv

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

def read_img(img_fn):
    #  return pil_img.fromarray(
                #  cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB))
    #  with open(img_fn, 'rb') as f:
        #  img = pil_img.open(f).convert('RGB')
    #  return img
    if img_fn.endswith('jpeg') or img_fn.endswith('jpg'):
        try:
            with open(img_fn, 'rb') as f:
                img = np.array(jpeg.JPEG(f).decode())
        except jpeg.JPEGRuntimeError:
            # logger.warning('{} produced a JPEGRuntimeError', img_fn)
            img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    else:
    #  elif img_fn.endswith('png') or img_fn.endswith('JPG') or img_fn.endswith(''):
        img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    return img

def get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2

def random_crop(center, scale, crop_scale_factor, axis='all'):
    '''
    center: bbox center [x,y]
    scale: bbox height / 200
    crop_scale_factor: amount of cropping to be applied
    axis: axis which cropping will be applied
        "x": center the y axis and get random crops in x
        "y": center the x axis and get random crops in y
        "all": randomly crop from all locations
    '''
    orig_size = int(scale * 200.)
    ul = (center - (orig_size / 2.)).astype(int)

    crop_size = int(orig_size * crop_scale_factor)

    if axis == 'all':
        h_start = np.random.rand()
        w_start = np.random.rand()
    elif axis == 'x':
        h_start = np.random.rand()
        w_start = 0.5
    elif axis == 'y':
        h_start = 0.5
        w_start = np.random.rand()
    else:
        raise ValueError(f'axis {axis} is undefined!')

    x1, y1, x2, y2 = get_random_crop_coords(
        height=orig_size,
        width=orig_size,
        crop_height=crop_size,
        crop_width=crop_size,
        h_start=h_start,
        w_start=w_start,
    )
    scale = (y2 - y1) / 200.
    center = ul + np.array([(y1 + y2) / 2, (x1 + x2) / 2])
    return center, scale

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def convert_crop_coords_to_orig_img_cliff(bbox, keypoints, crop_size):
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # unnormalize to crop coords
    keypoints[:,:,:2] = 0.5 * crop_size * (keypoints[:,:,:2] + 1.0)

    # rescale to orig img crop
    keypoints[:,:,:2] *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints[0]


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = resize(new_img, res)
    return new_img

def calculate_focal_length(img_h, img_w):
    return float((img_w**2 + img_h**2)**0.5)

def calculate_bbox_info(bb_center, bb_scale, orig_shape):

    img_h, img_w = orig_shape[0], orig_shape[1]
    cx, cy = bb_center[0], bb_center[1]
    b = bb_scale * 200
    focal_length = calculate_focal_length(img_h, img_w)

    bbox_info = np.array([cx - img_w / 2., cy - img_h / 2., b])

    # The constants below are used for normalization, and calculated from H36M data.
    bbox_info[:2] = bbox_info[:2] / focal_length * 2.8
    bbox_info[2] = (bbox_info[2] - 0.24 * focal_length) / (0.06 * focal_length)

    return bbox_info.astype(np.float32)

def crop_cv2(img, center, scale, res, rot=0):
    c_x, c_y = center
    c_x, c_y = int(round(c_x)), int(round(c_y))
    patch_width, patch_height = int(round(res[0])), int(round(res[1]))
    bb_width = bb_height = int(round(scale * 200.))

    trans = gen_trans_from_patch_cv(
        c_x, c_y, bb_width, bb_height,
        patch_width, patch_height,
        scale=1.0, rot=rot, inv=False,
    )

    crop_img = cv2.warpAffine(
        img, trans, (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    return crop_img


def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = scipy.misc.imresize(img, crop_shape, interp='nearest')
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    return new_img


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img


def flip_kp(kp):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = constants.J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:, 0] = - kp[:, 0]
    return kp


def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose

def rescale_cv2(img, rescale_fac):
    width = int(img.shape[1] * rescale_fac)
    height = int(img.shape[0] * rescale_fac)

    dsize = (width, height)
    img = cv2.resize(img, dsize, interpolation = cv2.INTER_LINEAR)
    return img


def generate_part_labels(vertices, faces, cam_t, K, R, dist_coeffs, body_part_texture, part_bins, neural_renderer):
    batch_size = vertices.shape[0]

    body_parts, depth, mask = neural_renderer(
        vertices,
        faces.expand(batch_size, -1, -1),
        textures=body_part_texture.expand(batch_size, -1, -1, -1, -1, -1),
        K=K.expand(batch_size, -1, -1),
        R=R.expand(batch_size, -1, -1),
        dist_coeffs=dist_coeffs,
        t=cam_t.unsqueeze(1),
    )

    render_rgb = body_parts.clone()

    body_parts = body_parts.permute(0, 2, 3, 1)
    body_parts *= 255. # multiply it with 255 to make labels distant
    body_parts, _ = body_parts.max(-1) # reduce to single channel

    body_parts = torch.bucketize(body_parts.detach(), part_bins, right=True) # np.digitize(body_parts, bins, right=True)

    # add 1 to make background label 0
    body_parts = body_parts.long() + 1
    body_parts = body_parts * mask.detach()

    return body_parts.long(), render_rgb

def get_body_part_texture(faces, n_vertices=6890, non_parametric=False):
    smpl_segmentation = joblib.load('data/smpl_partSegmentation_mapping.pkl')

    smpl_vert_idx = smpl_segmentation['smpl_index']
    nparts = 24.

    if non_parametric:
        # reduce the number of body_parts to 14
        # by mapping some joints to others
        nparts = 14.
        joint_mapping = map_smpl_to_common()

        for jm in joint_mapping:
            for j in jm[0]:
                smpl_vert_idx[smpl_vert_idx==j] = jm[1]

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, :3] = smpl_vert_idx[..., None]

    vertex_colors = color.to_rgba(vertex_colors)
    face_colors = vertex_colors[faces].min(axis=1)

    texture = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)
    texture[0, :, 0, 0, 0, :] = face_colors[:, :3] / nparts
    texture = torch.from_numpy(texture).float()
    return texture

def get_default_camera(focal_length, img_size):
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[2, 2] = 1
    K[0, 2] = img_size / 2.
    K[1, 2] = img_size / 2.
    K = K[None, :, :]
    R = torch.eye(3)[None, :, :]
    dist_coeffs = torch.FloatTensor([[0., 0., 0., 0., 0.,]])
    return K, R, dist_coeffs

def overlay_text(image, txt_str, str_id=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = image.shape[0]*0.0016
    thickness = int(image.shape[0]*0.005)
    bbox_offset = int(image.shape[0]*0.01)
    text_offset_x, text_offset_y = int(image.shape[1]*0.02), int(image.shape[0]*0.06*str_id)

    (text_width, text_height) = cv2.getTextSize(txt_str, font, fontScale=font_scale, thickness=thickness)[0]
    box_coords = ((text_offset_x, text_offset_y + bbox_offset), (text_offset_x + text_width + bbox_offset, text_offset_y - text_height - bbox_offset))

    cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(image, txt_str, (text_offset_x, text_offset_y), font, font_scale, (0, 0, 255), thickness)
    return image

def show_imgs(imgs, num_rows=1, size=15, live=False, legend=False, cmap=None, label=None,
              save_img=False, filename=None):
    if live == True:
        clear_output(wait=True)
    num_imgs_per_row = math.ceil(len(imgs)/num_rows)
    fig, axs = plt.subplots(num_rows, num_imgs_per_row, squeeze=False,
                            figsize=(size,size), constrained_layout=True)
    img_idx = 0
    for row in range(num_rows):
        for i in range(num_imgs_per_row):
            axs[row,i].imshow(imgs[img_idx])
            axs[row,i].axis('off')
            if img_idx < len(imgs) - 1:
                img_idx += 1
    if legend == True:
        patches = [mpatches.Patch(color=cmap[i],
                   label=label[i]) for i in cmap]
        plt.legend(handles=patches, loc=4,
                   borderaxespad=1, fontsize=8)
    if save_img == True:
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    else:
        plt.show()

def concat_images_np(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    assert imga.dtype == imgb.dtype, ''
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3)).astype(imga.dtype)
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images_np(image_np_list):
    """
    Combines N color images from a list of image ndarrays
    """
    output = None
    for i, img_np in enumerate(image_np_list):
        output = img_np if i==0 else contact_images_np(output, img)
    return output
