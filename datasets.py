import os
import torch
import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Dict, Any
from copy import deepcopy
from sklearn.datasets import make_moons
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


class MoonsDataset:

    def __init__(self, n_samples=2000, noise=0.1, random_state=0):
        self.X, self.y = make_moons(n_samples, noise, random_state)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.X.shape[0]

    def get_labels(self):
        return self.y


class DeprecatedDataset(Dataset):
    def __init__(self, table_name, mode='train'):
        # train, test, meta, cat_cols, ord_cols = load_dataset('adult', benchmark=True)
        # fm = CatBoostFeatureMaker(meta, cat_cols, ord_cols, sample=train.shape[0])
        # fm.fit(train)
        # train_x_arr, train_y_arr = fm.transform(train)
        # test_x_arr, test_y_arr = fm.transform(test)
        # fm = FeatureMaker(meta, sample=train.shape[0])
        # train_x_arr, train_y_arr = fm.make_features(train)
        # test_x_arr, test_y_arr = fm.make_features(test)
        # train_x, train_y = torch.from_numpy(train_x_arr.astype(np.float32)), torch.from_numpy(train_y_arr)
        # test_x, test_y = torch.from_numpy(test_x_arr), torch.from_numpy(test_y_arr)
        train_x = np.load('../SDGymBenchmarkData/dae/dae_adult.npy')
        train_y = np.load('../SDGymBenchmarkData/dae/dae_label.npy')
        test_x = train_x.copy()
        test_y = train_y.copy()
        train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
        test_x, test_y = torch.from_numpy(test_x), torch.from_numpy(test_y)
        self._train_dataset = TensorDataset(train_x, train_y)
        self._test_dataset = TensorDataset(test_x, test_y)
        self.mode = mode

    def features(self):
        return self.data.tensors[0].type(torch.float32)

    def labels(self):
        return self.data.tensors[1]

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        assert new_mode in ['train', 'test', 'valid']
        self._mode = new_mode
        self.data = self._train_dataset
        if self._mode == 'test':
            self.data = self._test_dataset

    def __getitem__(self, item):
        if self.mode == 'train':
            return self._train_dataset[item]
        elif self.mode == 'test':
            return self._test_dataset[item]




class UCIDataset(Dataset):

    @property
    def num_continuous(self):
        return len(self._continuous_cols)

    @property
    def num_categorical(self):
        return len(self._categorical_cols)

    @property
    def embeds(self) -> List[Tuple[int, int]]:
        return self._embeds

    @property
    def categorical_cols(self) -> List[int]:
        return self._categorical_cols

    @property
    def continuous_cols(self) -> List[int]:
        return self._continuous_cols

    def _read_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        self._categorical_cols = []
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


class AdultDataSet(UCIDataset):
    training_file = 'adult.trn.npz'
    test_file = 'adult.tst.npz'
    config_file = 'adult.json'

    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.scale = True
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

        return row[self.categorical_cols], row[self.continuous_cols], label


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
    def __init__(self, num_minority: int = 1000):
        self.num_minority = num_minority

    def __call__(self, dataset: UCIDataset):
        dataset = deepcopy(dataset)
        values, counts = np.unique(dataset.target, return_counts=True)
        min_idx = np.argmin(counts)
        minority_value = values[min_idx]
        minority_ids, _ = np.where(dataset.target == minority_value)
        majority_ids, _ = np.where(dataset.target != minority_value)
        num_minority_samples = min(minority_ids.size, self.num_minority)
        new_minority_ids = np.random.choice(minority_ids, num_minority_samples, replace=False)
        ids = np.concatenate([majority_ids, new_minority_ids])
        dataset.data = dataset.data[ids]
        dataset.target = dataset.target[ids]
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
