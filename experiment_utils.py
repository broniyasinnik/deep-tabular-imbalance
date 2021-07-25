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


class ExperimentLogger:

    def __init__(self, logdir: str, results_folder: str = "results"):
        self.logdir = logdir
        os.makedirs(os.path.join(self.logdir, results_folder), exist_ok=True)
        self.path_to_results = os.path.join(self.logdir, results_folder)
        self.loggers = {}

    def log_data(self, msg: str):
        if "data" not in self.loggers.keys():
            self.loggers["data"] = open(os.path.join(self.logdir, "data.txt"), "a+")
        self.loggers["data"].write(msg + "\n")

    def log_results(self, results: Dict[str, Any]):
        os.makedirs(self.path_to_results, exist_ok=True)
        if "predictions" in results:
            df = pd.DataFrame(data=results["predictions"])
            df.to_csv(os.path.join(self.logdir, "results/predictions.csv"),
                      index=False)
        if "pr_curve" in results:
            df = pd.DataFrame(data=results["pr_curve"])
            df.to_csv(os.path.join(self.logdir, "results/pr.csv"),
                      index=False)
            plt.figure()
            plt.step(df['recall'], df['precision'], where='post')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(
                'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                    .format(results["average_precision"]))
            plt.savefig(os.path.join(self.logdir, "results/pr.png"))
            plt.close()
        if "average_precision" in results:
            with open(os.path.join(self.logdir, "results/metrics.json"), 'w') as f:
                json.dump({'average_precision': results["average_precision"]}, f, indent=4)

    def close_log(self):
        for logger in self.loggers.values():
            logger.flush()
            logger.close()


class LoggingMode(Enum):
    OVERWRITE = 1
    DEBUG = 2


class experiment_logger:

    def __init__(self, path: str, name: str, mode: LoggingMode):
        self.mode = mode
        if self.mode == LoggingMode.DEBUG:
            log_to = os.path.join(path, "debug")
        else:
            log_to = os.path.join(path, name)
        self.logger = ExperimentLogger(logdir=log_to)

    def __enter__(self):
        if self.mode == LoggingMode.OVERWRITE or \
                self.mode == LoggingMode.DEBUG:
            if os.path.exists(self.logger.logdir):
                shutil.rmtree(self.logger.logdir)

        os.makedirs(self.logger.logdir, exist_ok=True)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close_log()


class ExperimentFactory:
    def __init__(self, train_file: str = None, test_file: str = None,
                 holdout_file: str = None, smote_file: str = None,
                 model=None, seed: float = None, hparams: Dict[str, Any] = None):
        self.train_file = train_file
        self.test_file = test_file
        self.smote_file = smote_file
        self.holdout_file = holdout_file
        self.hparams = hparams
        self.seed = seed

        self.model = model
        # self.criterion = {
        #     "bce": nn.BCEWithLogitsLoss(),
        #     "bce_weighted": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams["ir"]))
        # }
        self.criterion = nn.BCEWithLogitsLoss()
        if model is not None:
            self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.hparams["lr_model"])

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
            "criterion": self.criterion,
            "optimizer": self.optimizer
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
            "criterion": self.criterion,
            # "optimizer": self.optimizer
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
            "criterion": self.criterion,
            "optimizer": self.optimizer
        }
        return e_utils

    def prepare_base_experiment(self):
        train_data = TableDataset.from_npz(self.train_file, train=True)
        test_data = TableDataset.from_npz(self.test_file, train=False)
        ir = (train_data.target == 0).sum() / (train_data.target == 1).sum()
        self.model.classifier[-1].bias.data.fill_(np.log(1 / ir))
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], shuffle=True),
            "valid": DataLoader(test_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'base',
            "loaders": loaders,
            "criterion": self.criterion,
            "optimizer": self.optimizer
        }
        return e_utils

    def prepare_upsampling_experiment(self):
        train_data = TableDataset.from_npz(self.train_file, train=True)
        test_data = TableDataset.from_npz(self.test_file, train=False)
        upsampling = BalanceClassSampler(train_data.target.squeeze(), mode='upsampling')
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], sampler=upsampling),
            "valid": DataLoader(test_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'upsampling',
            "loaders": loaders,
            "criterion": self.criterion,
            "optimizer": self.optimizer,
        }
        return e_utils

    def prepare_downsampling_experiment(self):
        train_data = TableDataset.from_npz(self.train_file, train=True)
        test_data = TableDataset.from_npz(self.test_file, train=False)
        downsampling = BalanceClassSampler(train_data.target.squeeze(), mode='downsampling')
        loaders = {
            "train": DataLoader(train_data, batch_size=self.hparams['batch_size'], sampler=downsampling),
            "valid": DataLoader(test_data, batch_size=self.hparams['batch_size'], shuffle=False)
        }
        e_utils = {
            "name": 'downsampling',
            "loaders": loaders,
            "criterion": self.criterion,
            "optimizer": self.optimizer
        }
        return e_utils

    def prepare_meta_experiment_with_smote(self):
        smote_data = SyntheticDataset(data=self.train_file, synthetic_data=self.smote_file)
        test_data = TableDataset.from_npz(self.test_file, train=False)
        loaders = {
            "train": DataLoader(smote_data, batch_size=self.hparams['batch_size'], shuffle=True),
            "valid": DataLoader(test_data, batch_size=self.hparams['batch_size'], shuffle=False),
        }
        e_utils = {
            "name": 'meta',
            "loaders": loaders,
            "train_dataset": smote_data,
            "criterion": self.criterion,
            "optimizer": self.optimizer

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
