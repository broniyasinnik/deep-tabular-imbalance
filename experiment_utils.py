import functools
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import cycle
from typing import Any, Dict, List, Union

import catalyst
import numpy as np
import pandas as pd
import runners
import torch
import torch.nn as nn
import yaml
from catalyst.typing import Criterion, Model, Optimizer, Sampler, Scheduler
from catalyst.utils.torch import load_checkpoint
from catalyst import dl
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from datasets import TableSyntheticDataset, TableDataset
from models.net import Net

COLORS: List[str] = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'black', 'orange', 'blue']


def save_pr_curve(labels: np.array, scores: np.array, logdir: str):
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1.)
    thresholds = np.concatenate([thresholds, [1.]])
    df = pd.DataFrame(data={"precision": precision,
                            "recall": recall,
                            "thresholds": thresholds})
    assert os.path.exists(logdir), f"The directory {logdir} doesn't exist"
    df.to_csv(os.path.join(logdir, "pr.csv"),
              index=False)


def save_pr_figure(results, logdir):
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    colors = cycle(COLORS)
    for name in results:
        labels, scores = results.get(name)['labels'], results.get(name)['scores']
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        plt.step(recall, precision, color=next(colors), where='post', label=f"PR of {name} (area = {ap:.2f})")
    plt.legend()
    plt.savefig(os.path.join(logdir, "pr.png"))
    plt.close()


def save_roc_figure(results, logdir):
    plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve')
    colors = cycle(COLORS)
    for name in results:
        labels, scores = results.get(name)['labels'], results.get(name)['scores']
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.step(fpr, tpr, where='post', color=next(colors), label=f"ROC of {name} (area = {roc_auc:.2f})")
    plt.legend()
    plt.savefig(os.path.join(logdir, "roc.png"))
    plt.close()


def save_metrics_table(results: Dict[str, Any], save_ap: bool, save_auc: bool, p_at: List, logdir: str):
    metrics = defaultdict(list)
    for name in results:
        labels, scores = results.get(name)['labels'], results.get(name)['scores']
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        roc_auc = roc_auc_score(labels, scores)
        metrics["AP"].append(ap)
        metrics["AUC"].append(roc_auc)
        for r in p_at:
            metrics[f'P@{int(r * 100)}%'].append(precision[recall >= r][-1])

    df = pd.DataFrame(data=metrics, index=results.keys())
    df.to_csv(os.path.join(logdir, 'metrics.csv'), index=True)


def aggregate_results(results: Dict[str, Any], metrics: ConfigDict, logdir: str):
    if 'pr' in metrics:
        save_pr_figure(results, logdir)
    if 'roc' in metrics:
        save_roc_figure(results, logdir)
    save_ap = 'ap' in metrics
    save_auc = 'auc' in metrics
    p = next(filter(lambda x: x.startswith('p@'), metrics), [])
    if p:
        p = list(map(lambda x: float(x.strip()), p.partition('@')[2].split(',')))

    save_metrics_table(results, save_ap=save_ap, save_auc=save_auc, p_at=p, logdir=logdir)


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
    runner: dl.Runner = None
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

    def prepare_experiment(self, name: str):
        assert name in self.config.experiments, f"No available configuration for experiment {name}"
        conf_experiment = self.config.experiments[name]

        # train_data = TableDataset.from_npz(conf_experiment.datasets.train, train=True)
        train_data = get_train_data(train_file=conf_experiment.datasets.train,
                                    synth_file=conf_experiment.datasets.get('synthetic'))
        valid_data = TableDataset.from_npz(conf_experiment.datasets.valid, train=False)
        ir = (train_data.targets == 0).sum() / (train_data.targets == 1).sum()

        loaders = get_train_valid_loaders(train_data, valid_data, params=conf_experiment)
        model = get_model(self.config.model)
        if self.config.model.get("init_last_layer"):
            model.classifier[-1].bias.data.fill_(np.log(1 / ir))
        if model_path := conf_experiment.get("preload"):
            model.load_state_dict(load_checkpoint(model_path)['model_state_dict'])

        optimizer = get_optimizer(model, params=conf_experiment.get('optimizer'))
        scheduler = get_scheduler(optimizer,
                                  params=conf_experiment.get('scheduler'))
        criterion = get_criterion()
        runner = get_runner(conf_experiment.runner)
        experiment = Experiment(name=name, ir=ir, loaders=loaders, model=model, runner=runner,
                                optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                epochs=conf_experiment.epochs)
        return experiment

    # def prepare_meta_experiment(self, name: str):
    #     conf_experiment = self.config.experiments[name]
    #     train_data = SyntheticDataset(real_data=conf_experiment.datasets.train,
    #                                   synthetic_data=conf_experiment.datasets.synthetic_train,
    #                                   valid_data=conf_experiment.datasets.valid)
    #     valid_data = TableDataset.from_npz(conf_experiment.datasets.valid, train=False)
    #     loaders = get_train_valid_loaders(train_data, valid_data, params=conf_experiment)
    #     model = get_model(self.config.model)
    #     hparams = conf_experiment.hparams
    #     experiment = Experiment(name=name, loaders=loaders, model=model,
    #                             epochs=conf_experiment.epochs, hparams=hparams)
    #     return experiment


def check_empty_args(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        for arg in args:
            if arg is None:
                return None
        for k, v in kwargs.items():
            if v is None:
                return None

        return func(*args, **kwargs)

    return inner


def get_runner(params: ConfigDict):
    params = params.to_dict()
    _target = params.pop('_target_')
    runner = getattr(runners, _target)(**params)
    return runner


def get_train_data(train_file: str, synth_file: str = None):
    if synth_file is None:
        return TableDataset.from_npz(train_file, train=True)
    else:
        return TableSyntheticDataset(real_data=train_file,
                                     synthetic_data=synth_file)


def get_test_loader(test_file: str, batch_size: int = 128):
    test_data = TableDataset.from_npz(test_file, train=False)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return loader


def get_train_valid_loaders(train_data: Dataset, valid_data: Dataset, params: ConfigDict):
    sampler = get_sampler(train_data.targets.squeeze(), params.get("sampler"))
    shuffle = True if sampler is None else False
    loaders = {
        "train": DataLoader(train_data, batch_size=params.batch_size, shuffle=shuffle, sampler=sampler),
        "valid": DataLoader(valid_data, batch_size=params.batch_size, shuffle=False),
    }

    return loaders


def get_config(experiment_dir: str, config_name: str = 'config.yml') -> ConfigDict:
    conf_file = os.path.join(experiment_dir, config_name)
    assert os.path.exists(conf_file), "configuration file doesn't exist"
    config = load_config(conf_file)
    return config


def get_model(params: ConfigDict) -> Model:
    classifier = prepare_mlp_classifier(input_dim=params.input_dim, hidden_dims=params.hiddens)
    model = Net(classifier)
    return model


@check_empty_args
def get_optimizer(model, params: ConfigDict) -> Optimizer:
    opt_params = params.to_dict()
    _target = opt_params.pop('_target_')
    optimizer = getattr(torch.optim, _target)(model.classifier.parameters(), **opt_params)
    return optimizer


@check_empty_args
def get_scheduler(optimizer, params: ConfigDict) -> Scheduler:
    sch_params = params.to_dict()
    _target = sch_params.pop('_target_')
    scheduler = getattr(torch.optim.lr_scheduler, _target)(optimizer, **sch_params)

    return scheduler


@check_empty_args
def get_sampler(labels, params: ConfigDict) -> Sampler:
    samp_params = params.to_dict()
    _target = samp_params.pop('_target_')
    sampler = getattr(catalyst.data.sampler, _target)(labels=labels, **samp_params)
    return sampler


@check_empty_args
def get_criterion(pos_weight: float = 1.) -> Criterion:
    if pos_weight != 1.:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        return nn.BCEWithLogitsLoss()


@check_empty_args
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
