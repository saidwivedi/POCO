"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from loguru import logger
from .base_dataset import BaseDataset
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

class EFTMixedDataset(Dataset):

    def __init__(self, options, method, **kwargs):

        datasets_ratios = options.DATASETS_AND_RATIOS.split('_')
        num_ds = len(datasets_ratios)//2
        self.dataset_list = datasets_ratios[:num_ds]
        self.partition = [float(p) for p in datasets_ratios[num_ds:]]

        if len(self.dataset_list) > 1 and options.TRAIN_DS == 'stage':
            options.USE_SYNTHETIC_OCCLUSION = False

        assert(len(self.dataset_list) == len(self.partition), 'dataset and ratio does not match')

        logger.info(f'Datasets for training --> {self.dataset_list} with ratio {self.partition}')
        self.datasets = [eval(f'{options.DATASET_TYPE}')(options, method, ds, **kwargs) for ds in self.dataset_list]
        self.partition = np.array(self.partition).cumsum()
        self.length = max([len(ds) for ds in self.datasets])

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
