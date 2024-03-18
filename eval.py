import os
import torch
import pprint
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pocolib.core.trainer import LitModule
from pocolib.utils.os_utils import copy_code
from pocolib.utils.train_utils import load_pretrained_model
from pocolib.core.config import run_grid_search_experiments

def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')

    model = LitModule(hparams=hparams).to(device)

    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True)

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(
        gpus=1,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
    )

    logger.info('*** Started testing ***')
    trainer.test(model=model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
    parser.add_argument('--bid', type=int, default=300, help='amount of bid for cluster')
    parser.add_argument('--memory', type=int, default=16000, help='memory amount for cluster')
    parser.add_argument('--num_cpus', type=int, default=8, help='num cpus for cluster')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        script='train.py',
    )

    # from threadpoolctl import threadpool_limits
    # with threadpool_limits(limits=1):
    main(hparams)
