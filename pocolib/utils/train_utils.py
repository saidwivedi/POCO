import os
import re
import torch
import glob
import torch.nn as nn
import numpy as np
from loguru import logger
from collections import OrderedDict
import pytorch_lightning as pl
import torch.distributed as dist

from ..core.config import SMPL_MEAN_PARAMS

def get_index(arr, ele):
    try:
        return arr.index(ele)
    except ValueError:
        return None

def add_init_smpl_params_to_dict(state_dict):
    mean_params = np.load(SMPL_MEAN_PARAMS)
    init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
    init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
    init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
    state_dict['init_pose'] = init_pose
    state_dict['init_shape'] = init_shape
    state_dict['init_cam'] = init_cam
    return state_dict


def get_confident_frames(var, threshold):
    from .poco_utils import get_kinematic_uncert
    var = get_kinematic_uncert(var)

    min_uncert, max_uncert = var.min(1, keepdims=True), var.max(1, keepdims=1)
    # global_var = var - min_uncert / (max_uncert - min_uncert)
    # global_var = global_var[:,1:].mean(-1)
    # global_var[var[:,0] > 0.25] = 1.0

    global_var = var
    global_var = global_var[:,0]

    select_idx = (global_var < threshold).nonzero()[0]

    return select_idx



def auto_lr_finder(model, trainer):
    logger.info('Running auto learning rate finder')

    # Run learning rate finder
    lr_finder = trainer.lr_find(model)

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    logger.info(f'Found new learning rate {new_lr}')

    # update hparams of the model
    model.hparams.lr = new_lr

def set_seed(seed_value):
    if seed_value >= 0:
        logger.info(f'Seed value for the experiment {seed_value}')
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        pl.trainer.seed_everything(seed_value)

def prepare_statedict(state_dict):
    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if re.match('^model', name):
            name = name.replace('model.', '')
        if re.match('^backbone', name):
            name = name.replace('backbone.', '')
        if re.match('^head', name):
            name = name.replace('head.', '')
        if re.match('^uncert_head', name):
            name = name.replace('uncert_head.', '')
        if re.match('^flow_head', name):
            name = name.replace('flow_head.', '')
        new_state_dict[name] = param
    return new_state_dict

def get_part_statedict(full_statedict, part):
    part_statedict = {}
    for key in full_statedict.keys():
        if any(key.startswith(x) for x in [f'model.{part}.', f'{part}.']):
            part_statedict[key] = full_statedict[key]
    return prepare_statedict(part_statedict)

def freeze_model(model, freeze=False):
    for module in model.modules():
        if freeze == True:
            module.eval()
        else:
            module.train()
    for name, param in model.named_parameters():
        if 'bn' in name:
            param.requires_grad = not freeze
            param.track_running_stats = not freeze
        else:
            param.requires_grad = not freeze

def decode_freeze_params(freeze_params, max_epochs):
    if not freeze_params:
        return None
    freeze_stages = [i for i in freeze_params.split(',')]
    freeze_change = [int(fr_params.split('-')[0]) for fr_params in freeze_stages]
    freeze_change.append(max_epochs)
    freeze_dur = list(np.array(freeze_change[1:]) - np.array(freeze_change[:-1]))
    freeze_modules = [fr_params.split('-')[1:] for fr_params in freeze_stages]
    freeze_params_epoch = []
    for idx, fr_mod in enumerate(freeze_modules):
        freeze_params_epoch.extend([fr_mod]*freeze_dur[idx])
    return freeze_params_epoch

def load_statedict(model, full_statedict, modelname):
    state_dict = get_part_statedict(full_statedict, modelname)
    try:
        model.load_state_dict(state_dict, strict=True)
    except:
        logger.warning(f'Loading statedict for {modelname} in non-strict mode!!!')
        model.load_state_dict(state_dict, strict=False)

def get_model_path(path, inf_model='best'):
    if path.endswith('.pt') or path.endswith('.ckpt') or path.endswith('.pth'):
        return path
    else:
        if inf_model == 'best':
            return path + '/best_model.pt'
        elif inf_model == 'best_mpjpe_var':
            return path + '/best_mpjpe_var_model.pt'
        else:
            pt_file = sorted(glob.glob(f'{path}/tb_logs_poco-smpl/*/checkpoints/*'))[-1]
            return pt_file

def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True):
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        logger.warning(f'size mismatch for \"{pk}\": copying a param with shape {state_dict[pk].shape} '
                                       f'from checkpoint, the shape in current model is {model_state_dict[pk].shape}')
                        del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
