import os
import torch
from scipy.stats import rv_discrete, bernoulli
import numpy as np
import pandas as pd
import json
import random
from typing import List, Tuple, Dict, Any
from copy import deepcopy
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
                 noise: int = 0.2, transform=None, target_transform=None):
        self.train = train
        self.majority_samples = majority_samples
        self.minority_samples = minority_samples
        self.noise = noise
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            X, y = self.make_two_circles(seed=42)
            self.data = X
            self.target = y
        else:
            X, y = self.make_two_circles(seed=137)
            self.data = X
            self.target = y

    def make_two_circles(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        X, y = make_circles(n_samples=2 * self.majority_samples,
                            noise=self.noise)
        X0, y0 = X[y == 0], y[y == 0][:, None]
        X1, y1 = X[y==1][:self.minority_samples], y[y==1][:, None][:self.minority_samples]
        X0_in = np.random.multivariate_normal(mean=(0, 0), cov=((0.05, 0), (0, 0.05)),
                                           size=self.majority_samples)
        y0_in = np.zeros((self.majority_samples, 1))
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


class UCIDataset(Dataset):

    @property
    def embeds(self) -> List[Tuple[int, int]]:
        return self._embeds

    @property
    def categorical_cols(self) -> List[int]:
        return self._categorical_cols

    @property
    def continuous_cols(self) -> List[int]:
        return self._continuous_cols

    @property
    def categories_sizes(self) -> List[int]:
        return self._categories_sizes

    def _read_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        self._categorical_cols = []
        self._categories_sizes = []
        self._continuous_cols = []
        self._embeds = []
        for i, col in enumerate(config):
            if col.get('name') == 'label':
                self._label_col = i
            elif col.get('type') == 'continuous':
                self._continuous_cols.append(i)
            elif col.get('type') == 'categorical':
                self._categorical_cols.append(i)
                size, embed = col.get('size', 0), col.get('embed', 0)
                self._embeds.append((size, embed))
                self._categories_sizes.append(size)


class AdultDataSet(UCIDataset):
    training_file = 'adult.trn.npz'
    test_file = 'adult.tst.npz'
    config_file = 'adult.json'

    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        path_to_config = os.path.join(root, self.config_file)
        assert os.path.exists(path_to_config), "Config file doesn't exists"
        self._read_config(path_to_config)

        if self.train:
            path_to_file = os.path.join(root, self.training_file)
            assert os.path.exists(path_to_file), "Training file doesn't exists"
            data = np.load(path_to_file)
            self.data, self.target = data["train"], data["target"]
        else:
            path_to_file = os.path.join(root, self.test_file)
            assert os.path.exists(os.path.join(root, self.test_file)), "Test file doesn't exists"
            data = np.load(path_to_file)
            self.data, self.target = data["test"], data["target"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row, label = self.data[item], self.target[item]
        if self.transform is not None:
            row = self.transform(row)
        if self.target_transform is not None:
            label = self.target_transform(label)

        sample = {"input": row, "label": label}
        return sample


class ShuttleDataset(UCIDataset):
    # DataSet Constants #
    LABEL_COL = 9

    def __init__(self, train_file, test_file):
        train_df = pd.read_csv(train_file, sep=' ', header=None)
        test_df = pd.read_csv(test_file, sep=' ', header=None)
        self.train_x = train_df.drop(columns=self.LABEL_COL).to_numpy(dtype=np.float32)
        self.train_y = train_df[self.LABEL_COL].to_numpy(dtype=np.float32)
        self.test_x = test_df.drop(columns=self.LABEL_COL).to_numpy(dtype=np.float32)
        self.test_y = test_df[self.LABEL_COL].to_numpy(dtype=np.float32)
        # TODO: Check how to set up the validation set

    def scale_features(self, features: List, scale_type: str):
        scaler = FeatureScaler(scale_type)
        scaler.fit(self.train_x, features=features)
        self.train_x = scaler.transform(self.train_x)
        self.test_x = scaler.transform(self.test_x)

    @property
    def num_features(self):
        return self.train_x.shape[1]


class DatasetImbalanced:
    def __init__(self, imbalance_ratio: float = None, pos_label: int = 1):
        self.imbalance_ratio = imbalance_ratio
        self.pos_label = pos_label

    def __call__(self, dataset: UCIDataset, return_the_complement: bool = False):
        dataset = deepcopy(dataset)
        minority_ids, = np.where(dataset.target == self.pos_label)
        majority_ids, = np.where(dataset.target != self.pos_label)

        num_minority_samples = minority_ids.size
        num_majority_samples = majority_ids.size
        dataset_ir = num_minority_samples / num_majority_samples
        if self.imbalance_ratio is not None:
            assert 0 <= self.imbalance_ratio <= 1, "Imbalance Ratio should be between 0, 1"
            # rebalance the minority class
            if self.imbalance_ratio <= dataset_ir:
                num_minority = int(self.imbalance_ratio * majority_ids.size)
                num_minority_samples = min(minority_ids.size, num_minority)
            else:
                num_majority = int((1 / self.imbalance_ratio) * minority_ids.size)
                num_majority_samples = min(majority_ids.size, num_majority)

        new_minority_ids = np.random.choice(minority_ids, num_minority_samples, replace=False)
        new_majority_ids = np.random.choice(majority_ids, num_majority_samples, replace=False)
        ids = np.concatenate([new_majority_ids, new_minority_ids])
        if return_the_complement:
            complement_ids = np.setdiff1d(np.arange(dataset.data.shape[0]), ids)
            complement_dataset = deepcopy(dataset)
            complement_dataset.data = dataset.data[complement_ids]
            complement_dataset.target = dataset.target[complement_ids]
        dataset.data = dataset.data[ids]
        dataset.target = dataset.target[ids]
        pos_weight = num_majority_samples / num_minority_samples
        neg_weight = num_minority_samples / num_majority_samples
        setattr(dataset, "num_minority", num_minority_samples)
        setattr(dataset, "num_majority", num_majority_samples)
        setattr(dataset, "pos_weight", pos_weight)
        setattr(dataset, "neg_weight", neg_weight)

        if return_the_complement:
            return dataset, complement_dataset

        return dataset


class DatasetSSL(Dataset):

    def __init__(self, dataset: UCIDataset):
        self.dataset = dataset
        self.train_x_labeled = None
        self.train_y_labeled = None
        self.train_unlabeled = None
        self.test_x = self.dataset.test_x
        self.test_y = self.dataset.test_y

    def generate_labeled_unlabeled(self, num_labeled=1000, sample_classes: str = 'uniform'):
        '''

        :param num_labeled: Split the dataset with `num_labeled` labeled samples
        :param sample_classes: Either `uniform`,`minority` or `balanced` to sample from minority
        class only
        '''

        index = np.arange(self.dataset.train_y.shape[0])
        assert num_labeled <= index.size
        lbl_idx = index[:num_labeled]
        unlbl_idx = index[num_labeled:]

        if sample_classes == 'minority':
            labels, counts = np.unique(self.dataset.train_y, return_counts=True)
            majority_lbl = labels[counts.argmax()]
            minority_ids = index[self.dataset.train_y != majority_lbl]
            lbl_idx = minority_ids[:num_labeled]
            unlbl_idx = np.setdiff1d(index, lbl_idx)

        elif sample_classes == 'balanced':
            labels = np.unique(self.dataset.train_y)
            num_per_class = int(num_labeled / labels.size) + 1
            samples = []
            for label in labels:
                label_ids = index[self.dataset.train_y == label]
                label_ids = np.random.choice(label_ids, num_per_class)
                samples += label_ids.tolist()
            lbl_idx = np.array(samples)[:num_labeled]
            unlbl_idx = np.setdiff1d(index, lbl_idx)

        self.train_x_labeled = self.dataset.train_x[lbl_idx]
        self.train_y_labeled = self.dataset.train_y[lbl_idx]
        self.train_unlabeled = self.dataset.train_x[unlbl_idx]
