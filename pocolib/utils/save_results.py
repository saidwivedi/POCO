import os
import joblib

import torch
import numpy as np

from .eval_utils import calculate_distance_pose

class SaveResults():

    def __init__(self, hparams):
        super(SaveResults, self).__init__()
        self.METHOD = hparams.METHOD
        self.UNCERT_TYPE = hparams.POCO.UNCERT_TYPE
        self.SAVE_RESULTS = hparams.TESTING.SAVE_RESULTS
        self.NUM_GPUS = hparams.TRAINING.NUM_GPUS
        self.LOG_DIR = hparams.LOG_DIR
        self.VAL_DS = hparams.DATASET.VAL_DS
        self.sel_uncert_part = [x for x in range(24) if str(x) not in hparams.POCO.EXCLUDE_UNCERT_IDX.split('-')]

    def init_variables(self):

        self.evaluation_results = {
            'imgname': [],
            'dataset_name': [],
            'person_id': [],
            'mpjpe': [], # np.zeros((len(self.val_ds), 14)),
            'pampjpe': [], # np.zeros((len(self.val_ds), 14)),
            'v2v': [],
        }

        self.evaluation_results['corr_x'] = []
        self.evaluation_results['corr_y'] = []

        if self.SAVE_RESULTS:
            self.evaluation_results['pose'] = []
            self.evaluation_results['pred_jnts3D'] = []
            self.evaluation_results['gt_jnts3D'] = []
            self.evaluation_results['shape'] = []
            self.evaluation_results['cam_t'] = []
            if self.METHOD == 'poco':
                for un_type in self.UNCERT_TYPE:
                    self.evaluation_results[f'var_{un_type}'] = []

    def accumulate_metrics(self, err_jnt, r_err_jnt, v2v):

        if self.NUM_GPUS == 1:
            self.evaluation_results['mpjpe'] += err_jnt[:,:14].tolist()
            self.evaluation_results['pampjpe'] += r_err_jnt[:,:14].tolist()
            self.evaluation_results['v2v'] += v2v.tolist()

    def accumulate_preds(self, batch, pred):

        tolist = lambda x: [i for i in x.cpu().numpy()]
        if self.NUM_GPUS == 1:
            self.evaluation_results['imgname'] += batch['imgname']
            self.evaluation_results['dataset_name'] += batch['dataset_name']
            if 'person_id' in batch.keys():
                self.evaluation_results['person_id'] += tolist(batch['person_id'])

            if self.SAVE_RESULTS:
                # self.evaluation_results['pose'] += tolist(pred['pred_pose6d'])
                self.evaluation_results['pred_jnts3D'] += tolist(pred['pred_keypoints_3d'])
                self.evaluation_results['gt_jnts3D'] += tolist(batch['pose_3d'])
                self.evaluation_results['shape'] += tolist(pred['pred_shape'])
                self.evaluation_results['cam_t'] += tolist(pred['pred_cam_t'])
                if self.METHOD == 'poco':
                    for un_type in self.UNCERT_TYPE:
                        self.evaluation_results[f'var_{un_type}'] += tolist(pred[f'var_{un_type}'])

    def accumulate_corr_vect(self, pred, batch):

        dist = calculate_distance_pose(pred['pred_pose'], batch['pose'])[:,self.sel_uncert_part]
        uncert = pred['processed_uncert']
        # print(f'Distance --> ', dist)
        # print(f'uncert --> ', uncert)
        self.evaluation_results['corr_x'] += dist.cpu().numpy().flatten().tolist()
        self.evaluation_results['corr_y'] += uncert.flatten().tolist()

    def get_corr_vect(self):
        return np.array(self.evaluation_results['corr_x']), \
               np.array(self.evaluation_results['corr_y'])

    def dump_results(self, cur_epoch):

        if self.NUM_GPUS == 1:
            for k,v in self.evaluation_results.items():
                self.evaluation_results[k] = np.array(v)
            self.evaluation_results['epoch'] = cur_epoch
            joblib.dump(self.evaluation_results,
                        os.path.join(self.LOG_DIR,
                        f'evaluation_results_{self.VAL_DS}.pkl'))
