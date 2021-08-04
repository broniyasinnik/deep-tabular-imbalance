import pandas as pd
import torch
import os
import glob
from scipy.stats import rv_discrete
import numpy as np
import random
from typing import List, Union
from data_utils import load_arrays, sample_noisy_data
from catalyst.data.sampler import BalanceClassSampler
from sklearn.datasets import make_moons, make_circles
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, ConcatDataset



class MoonsDataset:

    def __init__(self, n_samples=2000, noise=0.1, random_state=0):
        self.X, self.y = make_moons(n_samples, noise, random_state)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.X.shape[0]

    def get_labels(self):
        return self.y


class CirclesDataSet(Dataset):

    def __init__(self, train: bool = True,
                 majority_samples: int = int(1e3), minority_samples: int = int(500),
                 noisy_minority_samples: int = 0, noisy_majority_samples: int = 0,
                 noise: int = 0.2, transform=None, target_transform=None):
        self.train = train
        self.majority_samples = majority_samples
        self.minority_samples = minority_samples
        self.noisy_minority_samples = noisy_minority_samples
        self.noisy_majority_samples = noisy_majority_samples
        self.noise = noise
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            X, y = self.make_two_circles(seed=42)
            self.data = X
            self.target = y
            self.synthetic_data = None
            self.sythetic_target = None
            if self.noisy_minority_samples > 0:
                noisy_minority = sample_noisy_data(self.noisy_minority_samples,
                                                   data=self.data[(self.target == 1).squeeze()])
                self.synthetic_data = noisy_minority
                self.synthetic_target = np.ones((self.noisy_minority_samples, 1))
            if self.noisy_majority_samples > 0:
                noisy_majority = sample_noisy_data(self.noisy_majority_samples,
                                                   data=self.data[(self.target == 0).squeeze()])

                if self.synthetic_data is None:
                    self.synthetic_data = noisy_majority
                    self.synthetic_target = np.zeros((self.noisy_majority_samples, 1))
                else:
                    self.synthetic_data = np.concatenate([self.synthetic_data, noisy_majority])
                    self.synthetic_target = np.concatenate([self.synthetic_target,
                                                            np.zeros((self.noisy_majority_samples, 1))])
        else:
            X, y = self.make_two_circles(seed=137)
            self.data = X
            self.target = y

    def make_two_circles(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        X, y = make_circles(n_samples=2 * self.majority_samples,
                            noise=self.noise)
        X0, y0 = X[y == 0][:self.majority_samples // 2], y[y == 0][:, None][:self.majority_samples // 2]
        X1, y1 = X[y == 1][:self.minority_samples], y[y == 1][:, None][:self.minority_samples]
        n_in_samples = self.majority_samples - self.majority_samples // 2
        X0_in = np.random.multivariate_normal(mean=(0, 0), cov=((0.05, 0), (0, 0.05)),
                                              size=n_in_samples)
        y0_in = np.zeros((n_in_samples, 1))
        X = np.concatenate([X0, X1, X0_in])
        y = np.concatenate([y0, y1, y0_in])
        return X, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        point, label = self.data[item], self.target[item]
        if self.transform is not None:
            point = self.transform(point)
        if self.target_transform is not None:
            label = self.target_transform(label)

        sample = {"features": point, "targets": label}
        return sample


class MixtureGaussiansDataset(Dataset):
    MEANS = np.array(
        [[-1, -4],
         [2, 3],
         [-3, 0],
         ])
    COVS = np.array(
        [[[1, 0.8], [0.8, 1]],
         [[1, -0.5], [-0.5, 1]],
         [[1, 0], [0, 1]],
         ])
    PROBS = np.array([
        0.2,
        0.5,
        0.3
    ])
    MEANS1 = np.array([
        [0, 0],
        # [-2, -2]
    ])

    COVS1 = np.array([
        [[0.1, 0], [0, 0.4]],
        # [[0.1, 0], [0, 0.1]]
    ])

    PROBS1 = np.array([
        1,
        # 0.5
    ])

    def __init__(self, majority=1000, minority=50):
        super(MixtureGaussiansDataset, self).__init__()
        data_majority = self.sample(majority, means=self.MEANS, covs=self.COVS, probs=self.PROBS)
        data_minority = self.sample(minority, means=self.MEANS1, covs=self.COVS1, probs=PROBS1)
        tensors_majority = TensorDataset(torch.tensor(data_majority), torch.zeros(majority, 1))
        tensors_minority = TensorDataset(torch.tensor(data_minority), torch.ones(minority, 1))
        dataset = ConcatDataset([tensors_majority, tensors_minority])
        self.dataset = dataset

    def sample(self, n, means, covs, probs):
        assert len(means) == len(covs) == len(probs), "number of components mismatch"
        components = len(means)
        comps_dist = rv_discrete(values=(range(components), probs))
        comps = comps_dist.rvs(size=n)
        conds = np.arange(components)[:, None] == comps[None, :]
        arr = np.array([np.random.multivariate_normal(means[c], covs[c], size=n)
                        for c in range(components)])
        return np.select(conds[:, :, None], arr).astype(np.float32)

    def __getitem__(self, item):
        return self.dataset[item]


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
        # features = None
        # targets = None
        # if isinstance(path_to_npz, str):
        #     path_to_npz = [path_to_npz]
        #
        # if isinstance(path_to_npz, list):
        #     for path in path_to_npz:
        #         assert os.path.exists(path), f"Training file {path} doesn't exists"
        #         data = np.load(path)
        #         X, y = torch.tensor(data["X"], dtype=torch.float32), \
        #           torch.tensor(data["y"].squeeze(), dtype=torch.float32)
        #         if features is not None:
        #             features = torch.cat((features, X))
        #         else:
        #             features = X
        #
        #         if targets is not None:
        #             targets = torch.cat((targets, y))
        #         else:
        #             targets = y

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
