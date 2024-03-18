import os

import shutil
import os.path as osp
from pathlib import Path
from loguru import logger
from distutils.dir_util import copy_tree

def copy_code(output_folder, curr_folder, code_folder='code'):
    code_folder = osp.join(output_folder, code_folder)
    if not osp.exists(code_folder):
        os.makedirs(code_folder)

    # Copy code
    logger.info('Copying main files ...')

    for f in [x for x in os.listdir(curr_folder) if x.endswith('.py')]:
        mainpy_path = osp.join(curr_folder, f)
        dest_mainpy_path = osp.join(code_folder, f)
        shutil.copy(mainpy_path, dest_mainpy_path)

    logger.info('Copying the rest of the source code ...')
    for f in ['pocolib']:
        src_folder = osp.join(curr_folder, f)
        dest_folder = osp.join(code_folder, osp.split(src_folder)[1])
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)
        copy_tree(src_folder, dest_folder)

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent
