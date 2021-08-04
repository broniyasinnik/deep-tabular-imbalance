import os
import shutil
import torch
import yaml
import numpy as np
import pandas as pd
import catalyst
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass, field
from ml_collections import ConfigDict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from catalyst.data.sampler import BalanceClassSampler
from catalyst.typing import Model, Optimizer, Scheduler, Criterion
from models.net import Net
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


def save_metrics(precision: np.array, recall: np.array, ap: float, auc: float, logdir: str):
    metrics = {"AP": [ap], "AUC": [auc]}
    for r in [0.25, 0.5, 0.75]:
        metrics[f'P@{int(r * 100)}%'] = [precision[recall >= r][-1]]
    df = pd.DataFrame(data=metrics)
    df.to_csv(os.path.join(logdir, 'metrics.csv'), index=False)


class LoggingMode(Enum):
    OVERWRITE = 1
    DEBUG = 2
    DISCARD = 3


class open_log:

    def __init__(self, path: str, name: str, mode: LoggingMode):
        self.mode = mode
        if self.mode == LoggingMode.DEBUG:
            log_to = os.path.join(path, "debug")
        else:
            log_to = os.path.join(path, name)

        self.logdir = log_to

    def __enter__(self):
        if self.mode == LoggingMode.OVERWRITE:
            if os.path.exists(self.logdir):
                shutil.rmtree(self.logdir)

        os.makedirs(self.logdir, exist_ok=True)
        return self.logdir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == LoggingMode.DISCARD:
            shutil.rmtree(self.logdir)


@dataclass
class Experiment:
    name: str = field(default='')
    ir: float = field(default=0.)
    epochs: int = field(default=0)
    loaders: Dict[str, torch.utils.data.DataLoader] = None
    model: Model = None
    optimizer: Optimizer = None
    scheduler: Scheduler = None
    criterion: Criterion = None
    hparams: Dict[str, Any] = None


class ExperimentFactory:
    def __init__(self, config: ConfigDict):
        self.config = config
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.valid_file = config.valid_file
        self.smote_file = config.smote_file if "smote_file" in config else None
        self.holdout_file = config.holdout_file if "holdout_file" in config else None
        # self.hparams = config.hparams
        self.seed = config.seed

    def get_test_loader(self):
        test_data = TableDataset.from_npz(self.test_file, train=False)
        loader = DataLoader(test_data, batch_size=self.hparams['batch_size'], shuffle=False)
        return loader

    def prepare_baseline_experiment(self, name: str):
        assert name in self.config.experiments, f"No available configuration for experiment {name}"
        conf_experiment = self.config.experiments[name]
        train_data = TableDataset.from_npz(conf_experiment.datasets.train, train=True)
        valid_data = TableDataset.from_npz(conf_experiment.datasets.valid, train=False)
        ir = (train_data.target == 0).sum() / (train_data.target == 1).sum()
        if 'sampler' in conf_experiment:
            sampler = prepare_sampler(train_data.target.squeeze(), conf_experiment.sampler)
            loaders = {
                "train": DataLoader(train_data, batch_size=conf_experiment.batch_size, sampler=sampler),
                "valid": DataLoader(valid_data, batch_size=conf_experiment.batch_size, shuffle=False)
            }
        else:
            loaders = {
                "train": DataLoader(train_data, batch_size=conf_experiment.batch_size, shuffle=True),
                "valid": DataLoader(valid_data, batch_size=conf_experiment.batch_size, shuffle=False)
            }
        model = prepare_model(self.config.model)
        optimizer = prepare_optimizer(model, params=conf_experiment.optimizer)
        scheduler = prepare_scheduler(optimizer,
                                      params=conf_experiment.scheduler) if 'scheduler' in conf_experiment else None
        criterion = prepare_criterion()
        experiment = Experiment(name=name, ir=ir, loaders=loaders, model=model,
                                optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                epochs=conf_experiment.epochs)
        return experiment

    def prepare_meta_experiment_with_smote(self, name: str):
        conf_experiment = self.config.experiments[name]
        smote_data = SyntheticDataset(real_data=conf_experiment.datasets.train,
                                      synthetic_data=conf_experiment.datasets.synthetic_train)
        valid_data = TableDataset.from_npz(conf_experiment.datasets.valid, train=False)
        loaders = {
            "train": DataLoader(smote_data, batch_size=conf_experiment.batch_size, shuffle=True),
            "valid": DataLoader(valid_data, batch_size=conf_experiment.batch_size, shuffle=False),
        }
        model = prepare_model(self.config.model)
        hparams = conf_experiment.hparams
        experiment = Experiment(name=name, loaders=loaders, model=model,
                                epochs=conf_experiment.epochs, hparams=hparams)
        return experiment


def prepare_config(experiment_dir: str, config_name: str = 'config.yml') -> ConfigDict:
    conf_file = os.path.join(experiment_dir, config_name)
    assert os.path.exists(conf_file), "configuration file doesn't exist"
    config = load_config(conf_file)
    return config


def prepare_model(params: ConfigDict):
    classifier = prepare_mlp_classifier(input_dim=params.input_dim, hidden_dims=params.hiddens)
    model = Net(classifier)
    return model


def prepare_optimizer(model, params: ConfigDict):
    opt_params = params.to_dict()
    _target = opt_params.pop('_target_')
    optimizer = getattr(torch.optim, _target)(model.classifier.parameters(), **opt_params)
    return optimizer


def prepare_scheduler(optimizer, params: ConfigDict):
    sch_params = params.to_dict()
    _target = sch_params.pop('_target_')
    scheduler = getattr(torch.optim.lr_scheduler, _target)(optimizer, **sch_params)

    return scheduler


def prepare_sampler(labels, params: ConfigDict):
    samp_params = params.to_dict()
    _target = samp_params.pop('_target_')
    sampler = getattr(catalyst.data.sampler, _target)(labels=labels, **samp_params)
    return sampler


def prepare_criterion(pos_weight: float = None):
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        return nn.BCEWithLogitsLoss()


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
