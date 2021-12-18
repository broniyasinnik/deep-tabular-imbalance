import pandas as pd
import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset
from copy import deepcopy
from typing import List, Tuple, Iterable, Union
from imblearn.over_sampling import SMOTE, ADASYN
from dataclasses import dataclass, field


class DatasetAgent:

    def __init__(self, dataset):
        self.__dict__["_dataset"] = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        return self._dataset[item]

    def __setattr__(self, key, value):
        if key in self._dataset.__dict__:
            setattr(self._dataset, key, value)
        else:
            super().__setattr__(key, value)

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __dir__(self) -> Iterable[str]:
        return dir(type(self)) + list(self.__dict__.keys()) + list(self._dataset.__dict__.keys())


@dataclass
class Column:
    name: str
    type: str
    max: float = None
    min: float = None
    size: int = None


class TableConfig:
    def __init__(self, config_path: str, label: str):
        self.path = config_path
        self._label = label
        self._read_config()

    @property
    def label_col(self) -> List[int]:
        return list(self._label_col.keys())

    @property
    def categorical_cols(self) -> List[int]:
        return list(self._categorical_cols.keys())

    @property
    def continuous_cols(self) -> List[int]:
        return list(self._continuous_cols.keys())

    @property
    def continuous_cols_ids(self) -> List[int]:
        return list(sorted(self._continuous_cols.values()))

    @property
    def categorical_cols_ids(self) -> List[int]:
        return list(self._categorical_cols.values())

    @property
    def categories_sizes(self) -> List[int]:
        return self._categories_sizes

    def _read_config(self):
        assert os.path.exists(self.path), "Config file doesn't exists!"
        with open(self.path) as f:
            config = json.load(f)
        self._categorical_cols = dict()
        self._label_col = dict()
        self._categories_sizes = dict()
        self._continuous_cols = dict()
        self.columns = []
        for i, col in enumerate(config):
            name = col.get('name')
            self.columns.append(Column(**col))
            if name == self._label:
                self._label_col[name] = i
            elif col.get('type') == 'continuous':
                self._continuous_cols[name] = i
            elif col.get('type') == 'categorical':
                size = col.get('size', 0)
                self._categorical_cols[name] = i
                self._categories_sizes[name] = size


def sample_noisy_data(n_samples: int, data: np.array, scale: float = 0.2):
    samples_ids = np.random.choice(data.shape[0], n_samples)
    samples = data[samples_ids]
    noisy_samples = samples + 0.2 * np.random.normal(scale=scale, size=samples.shape)
    return noisy_samples


def resample_with_smote(train_file: str):
    data_train = np.load(train_file)
    X, y = data_train["X"], data_train["y"]
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    X_resampled, y_resampled = X_resampled[X.shape[0]:], y_resampled[y.shape[0]:]
    assert X_resampled.shape[0] == y_resampled.shape[0]
    name = os.path.basename(train_file).partition('.')[0]
    name += '.smt.npz'
    smote_file = os.path.join(os.path.dirname(train_file), name)
    np.savez(smote_file, X=X_resampled, y=y_resampled)


def load_arrays(arrays: Union[str, List[str]]):
    if isinstance(arrays, str):
        arrays = [arrays]

    features = None
    targets = None
    if isinstance(arrays, list):
        for arr in arrays:
            assert os.path.exists(arr), f"Array in file {arr} doesn't exists"
            data = np.load(arr)
            X, y = data["X"].astype(np.float32), data["y"].astype(np.float32).squeeze()
            if features is not None:
                features = np.concatenate([features, X])
            else:
                features = X

            if targets is not None:
                targets = np.concatenate([targets, y])
            else:
                targets = y

    return features, targets


def create_imbalanced_dataset(x: Union[np.array, pd.DataFrame], y: np.array, ir: int, pos_label: int = 1,
                              random_state: int = 42):
    np.random.seed(random_state)
    minority_ids, = np.where(y.squeeze() == pos_label)
    majority_ids, = np.where(y.squeeze() != pos_label)

    num_minority_samples = minority_ids.size
    num_majority_samples = majority_ids.size
    dataset_ir = num_majority_samples / num_minority_samples

    # re-balance the minority class
    if ir >= dataset_ir:
        num_minority = int(1. / ir * majority_ids.size)
        num_minority_samples = min(minority_ids.size, num_minority)
    # re-balance the majority class
    else:
        num_majority = int(ir * minority_ids.size)
        num_majority_samples = min(majority_ids.size, num_majority)

    new_minority_ids = np.random.choice(minority_ids, num_minority_samples, replace=False)
    new_majority_ids = np.random.choice(majority_ids, num_majority_samples, replace=False)
    ids = np.concatenate([new_majority_ids, new_minority_ids])
    drop_ids = np.setdiff1d(np.arange(x.shape[0]), ids)
    if isinstance(x, np.ndarray):
        x_drop = x[drop_ids]
        y_drop = y[drop_ids]
        x_new = x[ids]
        y_new = y[ids]
    elif isinstance(x, pd.DataFrame):
        x_drop = x.iloc[drop_ids]
        y_drop = y[drop_ids]
        x_new = x.iloc[ids]
        y_new = y[ids]

    return {
        "X_imb": x_new,
        "y_imb": y_new,
        "X_drop": x_drop,
        "y_drop": y_drop
    }


class DatasetImbalanced(DatasetAgent):
    def __init__(self, dataset: Dataset, imbalance_ratio: float = None, pos_label: int = 1,
                 random_state: int = 42, create_complement_dataset: bool = False):
        super(DatasetImbalanced, self).__init__(deepcopy(dataset))
        self.imbalance_ratio = imbalance_ratio
        self.random_state = random_state
        self._pos_label = pos_label
        self._create_complement = create_complement_dataset
        self._create_imbalanced_dataset()

    def _create_imbalanced_dataset(self):
        np.random.seed(self.random_state)
        minority_ids, = np.where(self.target == self._pos_label)
        majority_ids, = np.where(self.target != self._pos_label)

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
        if self._create_complement:
            complement_ids = np.setdiff1d(np.arange(self.data.shape[0]), ids)
            complement_dataset = deepcopy(self._dataset)
            complement_dataset.data = complement_dataset.data[complement_ids]
            complement_dataset.target = complement_dataset.target[complement_ids]
        self.data = self.data[ids]
        self.target = self.target[ids]
        pos_weight = num_majority_samples / num_minority_samples
        neg_weight = num_minority_samples / num_majority_samples
        setattr(self, "num_minority", num_minority_samples)
        setattr(self, "num_majority", num_majority_samples)
        setattr(self, "pos_weight", pos_weight)
        setattr(self, "neg_weight", neg_weight)
        setattr(self, "ids", ids)

        if self._create_complement:
            setattr(self, "complement_dataset", complement_dataset)

    def save_npz(self, path: str):
        np.savez(path, X=self.data.numpy(), y=self.target.numpy())
