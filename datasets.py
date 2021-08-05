import torch
import numpy as np
import random
from typing import List, Union
from data_utils import load_arrays
from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data import Dataset


class TableDataset(Dataset):

    def __init__(self, features: Union[torch.Tensor, np.array], targets: Union[torch.Tensor, np.array],
                 train: bool = True, transform=None, target_transform=None):
        self.data = features
        self.target = targets
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    @classmethod
    def from_npz(cls, path_to_npz: Union[str, List[str]] = None, train: bool = True):
        features, targets = load_arrays(path_to_npz)
        self = cls(features, targets, train=train)
        return self

    def __repr__(self):
        descr = f'''{self.__class__.__name__} (shape {tuple(self.data.shape)}
        Majority size: {(self.target==0).sum()}
        Minority size: {(self.target==1).sum()}
        IR:{(self.target==0).sum()/(self.target==1).sum():.2f})
        '''
        return descr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row, label = self.data[item], self.target[item]
        if self.transform is not None:
            row = self.transform(row)
        if self.target_transform is not None:
            label = self.target_transform(label)

        if not isinstance(row, torch.Tensor):
            row = torch.tensor(row, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
        sample = {"features": row, "targets": label}
        return sample


class SyntheticDataset(Dataset):
    def __init__(self, real_data: Union[str, List[str]], synthetic_data: Union[str, List[str]]):

        # Features of the real data
        self._features_real, self._targets_real = load_arrays(real_data)

        # Features of the synthetic Data
        self._features_synthetic, self._targets_synthetic = load_arrays(synthetic_data)

        self.features = np.concatenate([self._features_real, self._features_synthetic])
        self.targets = np.concatenate([self._targets_real, self._targets_synthetic])
        self.size_real = len(self._targets_real)
        self.holdout_index = list(BalanceClassSampler(self._targets_real, mode="upsampling"))

    def __len__(self):
        return len(self.holdout_index)

    def __repr__(self):
        descr = f''' {self.__class__.__name__} shape {tuple(self.features.shape)}
        Number of real examples: {self.size_real} (#Maj {(self._targets_real == 0).sum()}, #Min {(self._targets_real == 1).sum()}) 
        Number of synthetic examples: {self._targets_synthetic.shape[0]}
        '''

        return descr

    def shuffle_holdout_index(self):
        random.shuffle(self.holdout_index)

    def get_real_dataset(self):
        return {'X': self.features[:self.size_real], 'y': self.targets[:self.size_real]}

    def get_synthetic_dataset(self):
        return {'X': self.features[self.size_real:], 'y': self.targets[self.size_real:]}

    def __getitem__(self, item):
        holdout_x = self._features_real[self.holdout_index[item]]
        holdout_y = self._targets_real[self.holdout_index[item]]
        x = self.features[item]
        y = self.targets[item]
        data_item = {
            "holdout_features": holdout_x,
            "holdout_target": holdout_y,
            "features": x,
            "target": y,
            "is_synthetic": np.array(False if item < self.size_real else True),
            "index": item
        }
        return data_item
