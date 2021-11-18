import functools
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union

import catalyst
import numpy as np
import torch
import torch.nn as nn
import yaml
from catalyst import dl, utils
from catalyst.typing import Criterion, Model, Optimizer, Sampler, Scheduler
from catalyst.utils.torch import load_checkpoint
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, Dataset
from runners import MetaClassificationRunner

import runners
from datasets import TableDataset
from models.net import Net


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
    name: str = field(default="")
    ir: float = field(default=0.0)
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
        # train_data = get_train_data(train_file=conf_experiment.datasets.train,
        #                             synth_file=conf_experiment.datasets.get('synthetic'))
        train_data = TableDataset.from_npz(conf_experiment.datasets.train, train=True,
                                           name=f"{name}_train")

        valid_data = TableDataset.from_npz(
            conf_experiment.datasets.valid, train=False, name=f"{name}_valid"
        )

        loaders = get_train_valid_loaders(train_data, valid_data, params=conf_experiment)
        model = get_model(self.config.model)
        if self.config.model.get("init_last_layer"):
            model.classifier[-1].bias.data.fill_(np.log(1 / train_data.ir))
        if model_path := conf_experiment.get("preload"):
            model.load_state_dict(load_checkpoint(model_path)["model_state_dict"])

        optimizer = get_optimizer(model, params=conf_experiment.get("optimizer"))
        scheduler = get_scheduler(optimizer, params=conf_experiment.get("scheduler"))
        criterion = get_criterion()
        runner = get_runner(conf_experiment.runner)
        if isinstance(runner, MetaClassificationRunner):
            synthetic_loader = get_loader(conf_experiment.datasets.synthetic,
                                          batch_size=conf_experiment.hparams.batch_synthetic,
                                          shuffle=True)
            holdout_loader = get_loader(conf_experiment.datasets.train_hold_out,
                                        batch_size=conf_experiment.hparams.batch_holdout,
                                        shuffle=True)
            runner.set_training_loaders(synthetic_loader, holdout_loader)
            if conf_experiment.datasets.get("valid_hold_out") is not None:
                ...

        experiment = Experiment(
            name=name,
            ir=train_data.ir,
            loaders=loaders,
            model=model,
            runner=runner,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            hparams=conf_experiment.get("hparams"),
            epochs=conf_experiment.epochs,
        )
        return experiment


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
    _target = params.pop("_target_")
    runner = getattr(runners, _target)(**params)
    return runner


def get_loader(data_file: str, batch_size: int = 128, shuffle: bool = False):
    test_data = TableDataset.from_npz(data_file, train=False)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_train_valid_loaders(train_data: Dataset, valid_data: Dataset, params: ConfigDict):
    sampler = get_sampler(train_data.targets.squeeze(), params.get("sampler"))
    shuffle = True if sampler is None else False
    loaders = {
        "train": DataLoader(
            train_data,
            batch_size=params.batch_size,
            shuffle=shuffle,
            sampler=sampler,
        ),
        "valid": DataLoader(valid_data, batch_size=params.batch_size, shuffle=False),
    }
    return loaders


def get_config(config_file: str = "config.yml") -> ConfigDict:
    assert os.path.exists(config_file), "configuration file doesn't exist"
    config = load_config(config_file)
    return config


def get_model(params: ConfigDict, checkpoint: str = None) -> Model:
    classifier = prepare_mlp_classifier(input_dim=params.input_dim, hidden_dims=params.hiddens)
    model = Net(classifier)
    if checkpoint:
        checkpoint = utils.load_checkpoint(path=checkpoint)
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=model,
        )
    return model


@check_empty_args
def get_optimizer(model, params: ConfigDict) -> Optimizer:
    opt_params = params.to_dict()
    _target = opt_params.pop("_target_")
    optimizer = getattr(torch.optim, _target)(model.classifier.parameters(), **opt_params)
    return optimizer


@check_empty_args
def get_scheduler(optimizer, params: ConfigDict) -> Scheduler:
    sch_params = params.to_dict()
    _target = sch_params.pop("_target_")
    scheduler = getattr(torch.optim.lr_scheduler, _target)(optimizer, **sch_params)

    return scheduler


@check_empty_args
def get_sampler(labels, params: ConfigDict) -> Sampler:
    samp_params = params.to_dict()
    _target = samp_params.pop("_target_")
    sampler = getattr(catalyst.data.sampler, _target)(labels=labels, **samp_params)
    return sampler


@check_empty_args
def get_criterion(pos_weight: float = 1.0) -> Criterion:
    if pos_weight != 1.0:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        return nn.BCEWithLogitsLoss()


@check_empty_args
def load_config(conf: str) -> ConfigDict:
    stream = open(conf, "r")
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
