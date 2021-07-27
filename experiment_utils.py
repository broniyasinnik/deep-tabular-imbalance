import os
import shutil
import torch
import yaml
import json
import numpy as np
import pandas as pd
import torch.nn as nn
from enum import Enum
from ml_collections import ConfigDict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from catalyst.data.sampler import BalanceClassSampler
from sklearn.model_selection import train_test_split
from datasets import SyntheticDataset
from datasets import TableDataset
from typing import Dict, Any, List, Union


def save_predictions(labels: np.array, scores: np.array, logdir: str):
    df = pd.DataFrame(data={"labels": labels,
                            "scores": scores})
    assert os.path.exists(logdir), f"The directory {logdir} doesn't exist"
    df.to_csv(os.path.join(logdir, "predictions.csv"),
              index=False)


def save_pr_curve(precision: np.array, recall: np.array, thresholds: np.array, ap: float, logdir: str):
    df = pd.DataFrame(data={"precision": precision,
                            "recall": recall,
                            "thresholds": thresholds})
    assert os.path.exists(logdir), f"The directory {logdir} doesn't exist"
    df.to_csv(os.path.join(logdir, "pr.csv"),
              index=False)
    plt.figure()
    plt.step(df['recall'], df['precision'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score: AP={0:0.2f}'.format(ap))
    plt.savefig(os.path.join(logdir, "pr.png"))
    plt.close()


def save_metrics(precision: np.array, recall: np.array, ap: float, logdir: str):
    metrics = {"AP": [ap]}
    for r in [0.25, 0.5, 0.75]:
        metrics[f'P@{int(r*100)}%'] = [precision[recall >= r][-1]]
    df = pd.DataFrame(data=metrics)
    df.to_csv(os.path.join(logdir, 'metrics.csv'), index=False)



class LoggingMode(Enum):
    OVERWRITE = 1
    DEBUG = 2


class open_log:

    def __init__(self, path: str, name: str, mode: LoggingMode):
        self.mode = mode
        if self.mode == LoggingMode.DEBUG:
            log_to = os.path.join(path, "debug")
        else:
            log_to = os.path.join(path, name)

        self.logdir = log_to
        # self.logger = ExperimentLogger(logdir=log_to)

    def __enter__(self):
        if self.mode == LoggingMode.OVERWRITE or \
                self.mode == LoggingMode.DEBUG:
            if os.path.exists(self.logdir):
                shutil.rmtree(self.logdir)

        os.makedirs(self.logdir, exist_ok=True)
        return self.logdir

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        # self.logger.close_log()


class ExperimentFactory:
    def __init__(self, config: ConfigDict):
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.valid_file = config.valid_file
        self.smote_file = config.smote_file if "smote_file" in config else None
        self.holdout_file = config.holdout_file if "holdout_file" in config else None
        self.hparams = config.hparams
        self.seed = config.seed

    def get_test_loader(self):
        test_data = TableDataset.from_npz(self.test_file, train=False)
        loader = DataLoader(test_data, batch_size=self.hparams['batch_size'], shuffle=False)
        return loader

    def prepare_potential_experiment(self):
        train_data = TableDataset.from_npz([self.train_file, self.holdout_file], train=True)
        test_data = TableDataset.from_npz(self.test_file, train=False)
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], shuffle=True),
            "valid": DataLoader(test_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'potential',
            "loaders": loaders,
        }
        return e_utils

    def prepare_optuna_experiment(self, validation_size: float = 0.2):
        data = np.load(self.train_file)
        X, y = data["X"], data["y"]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y,
                                                              test_size=validation_size, random_state=self.seed)
        train_data = TableDataset(features=X_train, targets=y_train)
        valid_data = TableDataset(features=X_valid, targets=y_valid)
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], shuffle=True),
            "valid": DataLoader(valid_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'optuna',
            "loaders": loaders,
        }
        return e_utils

    def prepare_smote_experiment(self):
        train_data = TableDataset.from_npz([self.train_file, self.smote_file], train=True)
        test_data = TableDataset.from_npz(self.test_file, train=False)
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], shuffle=True),
            "valid": DataLoader(test_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'smote',
            "loaders": loaders,
        }
        return e_utils

    def prepare_base_experiment(self):
        train_data = TableDataset.from_npz(self.train_file, train=True)
        valid_data = TableDataset.from_npz(self.valid_file, train=False)
        ir = (train_data.target == 0).sum() / (train_data.target == 1).sum()
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], shuffle=True),
            "valid": DataLoader(valid_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'base',
            "loaders": loaders,
            "ir": ir
        }
        return e_utils

    def prepare_upsampling_experiment(self):
        train_data = TableDataset.from_npz(self.train_file, train=True)
        valid_data = TableDataset.from_npz(self.valid_file, train=False)
        upsampling = BalanceClassSampler(train_data.target.squeeze(), mode='upsampling')
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], sampler=upsampling),
            "valid": DataLoader(valid_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'upsampling',
            "loaders": loaders,
        }
        return e_utils

    def prepare_downsampling_experiment(self):
        train_data = TableDataset.from_npz(self.train_file, train=True)
        valid_data = TableDataset.from_npz(self.valid_file, train=False)
        downsampling = BalanceClassSampler(train_data.target.squeeze(), mode='downsampling')
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], sampler=downsampling),
            "valid": DataLoader(valid_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'downsampling',
            "loaders": loaders,
        }
        return e_utils

    def prepare_meta_experiment_with_smote(self):
        smote_data = SyntheticDataset(data=self.train_file, synthetic_data=self.smote_file)
        valid_data = TableDataset.from_npz(self.valid_file, train=False)
        loaders = {
            "train": DataLoader(smote_data, batch_size=self.hparams['batch_size'], shuffle=True),
            "valid": DataLoader(valid_data, batch_size=self.hparams['batch_size'], shuffle=False),
        }
        e_utils = {
            "name": 'meta',
            "loaders": loaders,
            "train_dataset": smote_data,
        }
        return e_utils


def load_config(conf: str) -> ConfigDict:
    stream = open(conf, 'r')
    d = yaml.load(stream, Loader=yaml.FullLoader)
    return ConfigDict(d)


def prepare_mlp_classifier(input_dim: int, hidden_dims: Union[int, List[int]], output_dim: int = 1):
    layers = []
    hidden_dims = hidden_dims if type(hidden_dims) == list else [hidden_dims]
    inps = [input_dim] + hidden_dims
    outs = hidden_dims + [output_dim]

    for inp, out in zip(inps, outs):
        layers.append(nn.Linear(inp, out))
        if out != 1:
            layers.append(nn.ReLU())

    classifier = nn.Sequential(*layers)
    return classifier
