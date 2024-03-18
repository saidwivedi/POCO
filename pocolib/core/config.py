import os
import time
import yaml
import shutil
import argparse
import operator
import random
import itertools
from os.path import join
from loguru import logger
from functools import reduce
from yacs.config import CfgNode as CN
from typing import Dict, List, Union, Any
from flatten_dict import flatten, unflatten

from ..utils.cluster import execute_task_on_cluster

##### CONSTANTS #####
DATASET_NPZ_PATH = 'dataset_extras'

H36M_ROOT = 'dataset_folders/h36m'
LSP_ROOT = 'dataset_folders/lsp'
LSP_ORIGINAL_ROOT = 'dataset_folders/lsp-orig'
LSPET_ROOT = 'dataset_folders/hr-lspet'
MPII_ROOT = 'dataset_folders/mpii'
COCO_ROOT = 'dataset_folders/coco'
MPI_INF_3DHP_ROOT = 'dataset_folders/mpi_inf_3dhp'
PW3D_ROOT = 'dataset_folders/3dpw'
OH3D_ROOT = 'dataset_folders/3doh'
CHARADES_ROOT = 'dataset_folders/charades'
PASCAL_ROOT = '/ps/project/datasets/VOCdevkit/VOC2012'

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
COCO_OCCLUDERS_FILE = 'data/occlusion_augmentation/coco_train2014_occluders.pkl'
PASCAL_OCCLUDERS_FILE = 'data/occlusion_augmentation/pascal_occluders.pkl'

OPENPOSE_PATH = 'datasets/openpose'

DATASET_FOLDERS = {
    'h36m': H36M_ROOT,
    'h36m-p1': H36M_ROOT,
    'h36m-p2': H36M_ROOT,
    'lsp-orig': LSP_ORIGINAL_ROOT,
    'lsp': LSP_ROOT,
    'lspet': LSPET_ROOT,
    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
    'mpi-inf-3dhp-spin': MPI_INF_3DHP_ROOT,
    'mpii': MPII_ROOT,
    'coco': COCO_ROOT,
    'coco-cliff': COCO_ROOT,
    'coco-eft': COCO_ROOT,
    '3dpw': PW3D_ROOT,
    '3dpw-all': PW3D_ROOT,
    '3dpw-tr-ts': PW3D_ROOT,
    '3dpw-poco': PW3D_ROOT,
    '3doh': OH3D_ROOT,
    'charades': CHARADES_ROOT,

}

DATASET_FILES = [
    {
        '3dpw': '3dpw_test_with_mmpose.npz',
        '3doh': '3doh_test.npz',
    },
    {
        'h36m': 'h36m_train.npz',
        'mpii': 'mpii_train.npz',
        'coco': 'coco_2014_train.npz',
        'lspet': 'hr-lspet_train.npz',
        'mpi-inf-3dhp-spin': 'mpi_inf_3dhp_spin_train.npz',
        '3dpw': '3dpw_train.npz',
        '3doh': '3doh_train.npz',
        'charades': 'charades_train.npz',
    }
]

##### CONFIGS #####
hparams = CN()

# General settings
hparams.LOG_DIR = 'logs/experiments'
hparams.CONDOR_DIR = 'condor_logs'
hparams.METHOD = 'spin' # spin/poco
hparams.EXP_NAME = 'default'
hparams.EXP_ID = ''
hparams.RUN_TEST = False
hparams.SEED_VALUE = -1
hparams.PL_LOGGING = True
hparams.PREF_LOGGER = 'tensorboard' # tensorboard / tensorboard-comet

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.DATA_DIR = 'data'
hparams.DATASET.NOISE_FACTOR = 0.4
hparams.DATASET.ROT_FACTOR = 30
hparams.DATASET.FLIP = 1
hparams.DATASET.SCALE_FACTOR = 0.25
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 8
hparams.DATASET.PIN_MEMORY = True
hparams.DATASET.SHUFFLE_TRAIN = True
hparams.DATASET.SHUFFLE_VAL = False
hparams.DATASET.TRAIN_DS = 'all' # 'all'/'stage'/'coco'
hparams.DATASET.DATASETS_AND_RATIOS = 'h36m_coco_lspet_mpii_mpi-inf-3dhp-spin_0.5_0.233_0.046_0.021_0.2'
hparams.DATASET.STAGE_DATASETS = '0+h36m_1.0,1+h36m_coco_lspet_mpii_mpi-inf-3dhp-spin_0.5_0.233_0.046_0.021_0.2'
hparams.DATASET.DATASET_TYPE = 'BaseDataset' # BaseDataset, SMPLDataset
hparams.DATASET.VAL_DS = '3dpw'
hparams.DATASET.NUM_IMAGES = -1
hparams.DATASET.IMG_RES = 224
hparams.DATASET.FOCAL_LENGTH = 5000.
hparams.DATASET.IGNORE_3D = False
hparams.DATASET.RESCALE_FAC = 0.224
hparams.DATASET.MESH_COLOR = 'light_pink'
hparams.DATASET.DATA_TYPE = 'eft_data'
hparams.DATASET.MIXED_TYPE = 'EFTMixed'
hparams.DATASET.GENDER_EVAL = True
hparams.DATASET.USE_SYNTHETIC_OCCLUSION = False
hparams.DATASET.OCC_AUG_DATASET = 'pascal'
hparams.DATASET.UNCERT_THRESHOLD = 0.3

# optimizer config
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 0.0001
hparams.OPTIMIZER.WD = 0.0
hparams.OPTIMIZER.MM = 0.9
hparams.OPTIMIZER.AMSGRAD = False
hparams.OPTIMIZER.MODULE_LR = '' # 0.1_0.1_1.0_1.0

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.RESUME = None
hparams.TRAINING.PRETRAINED = None
hparams.TRAINING.PRETRAINED_LIT = None
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.LOG_SAVE_INTERVAL = 40
hparams.TRAINING.LOG_FREQ_TB_IMAGES = 500
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.FREEZE_PARAMS = '' # '0-backbone-head,1-flow_head,2'
hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH = True
hparams.TRAINING.SAVE_IMAGES = False
hparams.TRAINING.USE_AUGM = True
hparams.TRAINING.USE_SMPL_RENDER_LOSS = False
hparams.TRAINING.USE_SMPL_SEGM_LOSS = False

hparams.TRAINING.DIST_BACK = 'ddp'
hparams.TRAINING.NUM_GPUS = 1
hparams.TRAINING.PRECISION = 32
hparams.TRAINING.GRAD_CLIP_VAL = 0.

# Training process hparams
hparams.TESTING = CN()
hparams.TESTING.SAVE_IMAGES = False
hparams.TESTING.SAVE_RESULTS = False
hparams.TESTING.SIDEVIEW = True
hparams.TESTING.LOG_FREQ_TB_IMAGES = 50
hparams.TESTING.DISP_ALL = True
hparams.TESTING.DATASET_TYPE = 'BaseDataset' # TestAugDataset, BaseDataset
hparams.TESTING.TEST_ROT = 0
hparams.TESTING.TEST_SCALE = 1.0
hparams.TESTING.INF_MODEL = 'best' # best/last

# SPIN method hparams
hparams.SPIN = CN()
hparams.SPIN.BACKBONE = 'resnet50'

hparams.SPIN.SHAPE_LOSS_WEIGHT = 0.0
hparams.SPIN.KEYPOINT_3D_LOSS_WEIGHT = 5.
hparams.SPIN.KEYPOINT_2D_LOSS_WEIGHT = 2.5
hparams.SPIN.KEYPOINT_2D_NONCROP = False
hparams.SPIN.POSE_LOSS_WEIGHT = 1.
hparams.SPIN.BETA_LOSS_WEIGHT = 0.001
hparams.SPIN.OPENPOSE_TRAIN_WEIGHT = 0.
hparams.SPIN.GT_TRAIN_WEIGHT = 1.
hparams.SPIN.LOSS_WEIGHT = 60.
hparams.SPIN.SMPL_RENDER_LOSS_WEIGHT = 1.
hparams.SPIN.SMPL_SEGM_LOSS_WEIGHT = 1.

# POCO method hparams
hparams.POCO = CN()
hparams.POCO.BACKBONE = 'resnet50'
hparams.POCO.ACTIVATION_TYPE = 'sigmoid'
hparams.POCO.UNCERT_TYPE = 'pose' # pose-betas-joints
hparams.POCO.UNCERT_LAYER = 'diff_branch' # same_branch_v1 / diff_branch / diff_branch_lc2d
hparams.POCO.UNCERT_INP_TYPE = 'feat' # feat / feat-pose
hparams.POCO.KINEMATIC_UNCERT = False
hparams.POCO.NUM_NEURONS = ''
hparams.POCO.NUM_FLOW_LAYERS = 3
hparams.POCO.SIGMA_DIM = 9
hparams.POCO.NUM_NF_RV = 9
hparams.POCO.MASK_PARAMS_ID = ''
hparams.POCO.NFLOW_MASK_TYPE = 'alter'
hparams.POCO.EXCLUDE_UNCERT_IDX = ''
hparams.POCO.USE_DROPOUT = True
hparams.POCO.USE_ITER_FEATS = True
hparams.POCO.COND_NFLOW = False
hparams.POCO.CONTEXT_DIM = 1024
hparams.POCO.GT_POSE_COND = False
hparams.POCO.GT_POSE_COND_DS = 'h36m'
hparams.POCO.GT_POSE_COND_RATIO = 0.25

hparams.POCO.SHAPE_LOSS_WEIGHT = 0.0
hparams.POCO.KEYPOINT_3D_LOSS_WEIGHT = 5.
hparams.POCO.KEYPOINT_2D_LOSS_WEIGHT = 2.5
hparams.POCO.KEYPOINT_2D_NONCROP = False
hparams.POCO.POSE_LOSS_WEIGHT = 1.
hparams.POCO.BETA_LOSS_WEIGHT = 0.001
hparams.POCO.OPENPOSE_TRAIN_WEIGHT = 0.
hparams.POCO.GT_TRAIN_WEIGHT = 1.
hparams.POCO.POSE_UNCERT_WEIGHT = 1.
hparams.POCO.BETA_UNCERT_WEIGHT = 1.
hparams.POCO.JNT_UNCERT_WEIGHT = 1.
hparams.POCO.NF_LOSS_WEIGHT = 1.
hparams.POCO.GENG_LOSS_WEIGHT = 1.
hparams.POCO.USE_KEYCONF = False
hparams.POCO.LOSS_WEIGHT = 60.
hparams.POCO.LOSS_VER = 'norm_flow_res_gauss' # norm_flow_res, norm_flow_res_gaus
hparams.POCO.UNCERT_STATS_FILE = ''
hparams.POCO.SMPL_RENDER_LOSS_WEIGHT = 1.
hparams.POCO.SMPL_SEGM_LOSS_WEIGHT = 1.

hparams.POCO.LOG_TRAIN_UNCERT = 100
hparams.POCO.LOG_UNCERT_STAT = 5

def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()


def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()


def get_grid_search_configs(config, excluded_keys=[]):
    """
    :param config: dictionary with the configurations
    :return: The different configurations
    """

    def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
        """
        boolean to string conversion
        :param x: list or bool to be converted
        :return: string converted thinghat
        """
        if isinstance(x, bool):
            return [str(x)]
        for i, j in enumerate(x):
            x[i] = str(j)
        return x

    # exclude from grid search

    flattened_config_dict = flatten(config, reducer='path')
    hyper_params = []

    for k,v in flattened_config_dict.items():
        if isinstance(v,list):
            if k in excluded_keys:
                flattened_config_dict[k] = ['+'.join(v)]
            elif len(v) > 1:
                hyper_params += [k]

        if isinstance(v, list) and isinstance(v[0], bool) :
            flattened_config_dict[k] = bool_to_string(v)

        if not isinstance(v,list):
            if isinstance(v, bool):
                flattened_config_dict[k] = bool_to_string(v)
            else:
                flattened_config_dict[k] = [v]

    keys, values = zip(*flattened_config_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_id, exp in enumerate(experiments):
        for param in excluded_keys:
            exp[param] = exp[param].strip().split('+')
        for param_name, param_value in exp.items():
            # print(param_name,type(param_value))
            if isinstance(param_value, list) and (param_value[0] in ['True', 'False']):
                exp[param_name] = [True if x == 'True' else False for x in param_value]
            if param_value in ['True', 'False']:
                if param_value == 'True':
                    exp[param_name] = True
                else:
                    exp[param_name] = False


        experiments[exp_id] = unflatten(exp, splitter='path')

    return experiments, hyper_params


def run_grid_search_experiments(
        cfg_id,
        cfg_file,
        use_cluster,
        bid,
        memory,
        exclude_nodes,
        script='main.py',
        gpu_min_mem=10000,
):
    cfg = yaml.load(open(cfg_file))

    # parse config file to get a list of configs and related hyperparameters
    different_configs, hyperparams = get_grid_search_configs(
        cfg,
        excluded_keys=[],
    )
    logger.info(f'Grid search hparams: \n {hyperparams}')

    different_configs = [update_hparams_from_dict(c) for c in different_configs]
    logger.info(f'======> Number of experiment configurations is {len(different_configs)}')
    config_to_run = CN(different_configs[cfg_id])

    logger.info(f'===> Number of GPUs {config_to_run.TRAINING.NUM_GPUS}')

    if use_cluster:
        cls_run_folder = os.path.join('scripts/', config_to_run.METHOD)
        new_cfg_file = os.path.join(cls_run_folder, f'{config_to_run.EXP_NAME}_config.yaml')
        os.makedirs(cls_run_folder, exist_ok=True)
        shutil.copy(src=cfg_file, dst=new_cfg_file)
        execute_task_on_cluster(
            script=script,
            exp_name=config_to_run.EXP_NAME,
            method=config_to_run.METHOD,
            num_exp=len(different_configs),
            cfg_file=new_cfg_file,
            bid_amount=bid,
            num_workers=config_to_run.DATASET.NUM_WORKERS * config_to_run.TRAINING.NUM_GPUS,
            memory=memory,
            exclude_nodes=exclude_nodes,
            gpu_min_mem=gpu_min_mem,
            num_gpus=config_to_run.TRAINING.NUM_GPUS,
        )
        exit()

    # ==== create logdir using hyperparam settings
    if config_to_run.TRAINING.NUM_GPUS > 1:
        logtime = time.strftime('%d-%m-%Y_%H-%M')[:-1]
    else:
        logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{config_to_run.EXP_NAME}_ID{cfg_id:02d}_{logtime}'
    config_to_run.EXP_ID += f'{config_to_run.EXP_NAME}_ID{cfg_id:02d}'

    def get_from_dict(dict, keys):
        return reduce(operator.getitem, keys, dict)

    exp_id = ''
    for hp in hyperparams:
        v = get_from_dict(different_configs[cfg_id], hp.split('/'))
        exp_id += f'{hp.replace("/", ".").replace("_", "").lower()}-{v}'
    exp_id = exp_id.replace('/', '.')

    if exp_id:
        logdir += f'_{exp_id}'
        config_to_run.EXP_ID += f'/{exp_id}'

    logdir = os.path.join(config_to_run.LOG_DIR, config_to_run.METHOD, config_to_run.EXP_NAME, logdir)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=os.path.join(config_to_run.LOG_DIR, 'config.yaml'))

    config_to_run.LOG_DIR = logdir

    def save_dict_to_yaml(obj, filename, mode='w'):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save config
    save_dict_to_yaml(
        unflatten(flatten(config_to_run)),
        os.path.join(config_to_run.LOG_DIR, 'config_to_run.yaml')
    )

    # Add random sleep to avoid bursting request to proxy server
    # if cfg_id > 0 and config_to_run.PL_LOGGING:
    #     rnd_sleep = random.randint(0, 60)
    #     logger.info(f'Sleeping for {rnd_sleep} seconds to avoid bursting requests')
    #     time.sleep(rnd_sleep)

    return config_to_run
