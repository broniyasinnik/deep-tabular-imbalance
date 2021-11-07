import torch
import numpy as np
from typing import List, Union
from data_utils import load_arrays
from torch.utils.data import Dataset


class TableDataset(Dataset):
    def __init__(
            self,
            features: Union[torch.Tensor, np.array],
            targets: Union[torch.Tensor, np.array],
            train: bool = True,
            transform=None,
            target_transform=None,
    ):
        self.features = features
        self.targets = targets
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    @classmethod
    def from_npz(cls, path_to_npz: Union[str, List[str]] = None, train: bool = True):
        features, targets = load_arrays(path_to_npz)
        self = cls(features, targets, train=train)
        return self

    def __repr__(self):
        descr = f"""{self.__class__.__name__} (shape {tuple(self.features.shape)}
        Majority size: {(self.targets == 0).sum()}
        Minority size: {(self.targets == 1).sum()}
        IR:{(self.targets == 0).sum() / (self.targets == 1).sum():.2f})
        """
        return descr

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        row, label = self.features[item], self.targets[item]
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


class TableSyntheticDataset(Dataset):
    def __init__(
            self,
            real_data: Union[str, List[str]],
            synthetic_data: Union[str, List[str]],
    ):
        # Features of the real data
        self.features, self.targets = load_arrays(real_data)

        # Features of the synthetic Data
        self.features_synthetic, self.targets_synthetic = load_arrays(synthetic_data)

        # Features of the holdout data
        # self._features_holdout, self._targets_holdout = load_arrays(valid_data)

        # self.features = np.concatenate([self._features_real, self._features_synthetic])
        # self.target = np.concatenate([self._targets_real, self._targets_synthetic])

        self.size_real = len(self.targets)
        self.size_synth = len(self.targets_synthetic)
        # self.size_holdout = len(self._targets_holdout)

    def __len__(self):
        return self.size_synth

    def __repr__(self):
        descr = f""" {self.__class__.__name__} shape {tuple(self.features.shape)}
        Number of real examples: {self.size_real} (#Maj {(self.targets == 0).sum()}, #Min {(self.targets == 1).sum()})
        Number of synthetic examples: {self.targets_synthetic.shape[0]}
        """

        return descr

    @property
    def real_dataset(self):
        return {"X": self.features[: self.size_real], "y": self.target[: self.size_real]}

    @property
    def synthetic_dataset(self):
        return {"X": self.features[self.size_real:], "y": self.target[self.size_real:]}

    def __getitem__(self, item):
        # holdout_x = self._features_holdout[item % self.size_holdout]
        # holdout_y = self._targets_real[item % self.size_holdout]
        # x = self.features[item]
        # y = self.target[item]
        # data_item = {
        #     "holdout_features": holdout_x,
        #     "holdout_target": holdout_y,
        #     "features": x,
        #     "target": y,
        #     "is_synthetic": np.array(False if item < self.size_real else True),
        #     "index": item,
        # }
        t = item % 2
        rand_ind = np.random.randint(self.features[self.targets == t].shape[0])
        data_item = {
            "features_z": self.features_synthetic[item],
            "targets_z": self.targets_synthetic[item],
            "features_x": self.features[self.targets == t][rand_ind],
            "targets_x": self.targets[self.targets == t][rand_ind],
            "item": item
        }
        return data_item
