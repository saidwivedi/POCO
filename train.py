import os
import torch
import pprint
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from pocolib.core.trainer import LitModule
from pocolib.utils.os_utils import copy_code
from pocolib.utils.train_utils import load_pretrained_model, set_seed
from pocolib.core.config import run_grid_search_experiments

def main(hparams):

    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(hparams.SEED_VALUE)

    # Scale batch_size, num_workers, LR depending on num of gpus available
    num_gpus = hparams.TRAINING.NUM_GPUS
    hparams.DATASET.NUM_WORKERS = min(8, hparams.DATASET.NUM_WORKERS*num_gpus)
    hparams.OPTIMIZER.LR *= num_gpus

    if hparams.DATASET.TRAIN_DS == 'stage':
        hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH = True

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    if hparams.PL_LOGGING == True:
        copy_code(
            output_folder=log_dir,
            curr_folder=os.path.dirname(os.path.abspath(__file__)))

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')

    model = LitModule(hparams=hparams).to(device)

    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True)

    ckpt_callback = None
    logger_list = None

    # Turn on PL logging and Checkpoint saving
    if hparams.PL_LOGGING == True:
        ckpt_callback = ModelCheckpoint(
            monitor='val_loss',
            verbose=True,
            save_top_k=5,
            mode='min',
        )

        # Initialize comet cloud logger
        comet_logger = None
        # comet_logger = CometLogger(
        #     api_key='',
        #     workspace='',
        #     project_name='',
        #     experiment_name=hparams.EXP_ID
        # )

        # initialize tensorboard logger
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name='tb_logs',
        )
        logger_list = [tb_logger, comet_logger]
        device_stats = DeviceStatsMonitor() # Not logging for now


    trainer = pl.Trainer(
        gpus=num_gpus,
        precision=hparams.TRAINING.PRECISION,
        gradient_clip_val=hparams.TRAINING.GRAD_CLIP_VAL,
        strategy=hparams.TRAINING.DIST_BACK if num_gpus > 1 else None,
        logger=logger_list,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        log_every_n_steps=hparams.TRAINING.LOG_SAVE_INTERVAL,
        default_root_dir=log_dir,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        enable_checkpointing=hparams.PL_LOGGING,
        callbacks=[ckpt_callback] if hparams.PL_LOGGING else [],
        reload_dataloaders_every_n_epochs=hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        num_sanity_val_steps=0,
    )

    if hparams.RUN_TEST:
        logger.info('*** Started testing ***')
        trainer.test(model=model)
    else:
        logger.info('*** Running full validation first ***')
        trainer.validate(model)
        logger.info('*** Started training ***')
        trainer.fit(model)
        trainer.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
    parser.add_argument('--bid', type=int, default=5, help='amount of bid for cluster')
    parser.add_argument('--memory', type=int, default=64000, help='memory amount for cluster')
    parser.add_argument('--gpu_min_mem', type=int, default=20000, help='minimum gpu mem')
    parser.add_argument('--exclude_nodes', type=str, default='', help='exclude the nodes from submitting')
    parser.add_argument('--num_cpus', type=int, default=8, help='num cpus for cluster')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        exclude_nodes=args.exclude_nodes,
        script='train.py',
        gpu_min_mem=args.gpu_min_mem,
    )
    main(hparams)
