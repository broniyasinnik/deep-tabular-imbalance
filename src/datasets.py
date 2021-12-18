from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import load_arrays


class TableDataset(Dataset):
    def __init__(
        self,
        features: Union[torch.Tensor, np.array],
        targets: Union[torch.Tensor, np.array],
        train: bool = True,
        name: Optional[str] = None,
        transform=None,
        target_transform=None,
    ):
        self.features = features
        self.targets = targets
        self.name = name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    @classmethod
    def from_npz(
        cls,
        path_to_npz: Union[str, List[str]] = None,
        train: bool = True,
        name: Optional[str] = None,
    ):
        features, targets = load_arrays(path_to_npz)
        self = cls(features, targets, train=train, name=name)
        return self

    @property
    def ir(self):
        majority = (self.targets == 0).sum()
        minority = (self.targets == 1).sum()
        if minority > 0:
            return majority / minority
        else:
            return np.Inf

    def __repr__(self):
        descr = f"""{self.__class__.__name__} (shape {tuple(self.features.shape)}
        Majority size: {(self.targets == 0).sum()}
        Minority size: {(self.targets == 1).sum()}
        IR:{self.ir:.2f})
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
        sample = {"features": row, "targets": label, "index": item}
        return sample
