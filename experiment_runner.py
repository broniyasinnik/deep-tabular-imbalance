import os
import io
import torch
import torch.nn as nn
import numpy as np
import contextlib
from catalyst import utils
from collections import OrderedDict
from models.metrics import APMetric, PRMetric
from catalyst.utils.misc import set_global_seed
from callabacks import LogPRCurve, SaveSyntheticData
from catalyst import dl
from ml_collections import ConfigDict
from runners import ClassificationRunner, MetaClassificationRunner, evaluate_model
from experiment_utils import open_log, LoggingMode
from models.net import Net
from experiment_utils import ExperimentFactory, load_config, prepare_mlp_classifier


def prepare_config(experiment_dir: str) -> ConfigDict:
    conf_file = os.path.join(experiment_dir, 'config.yml')
    assert os.path.exists(conf_file), "configuration file doesn't exist"
    config = load_config(conf_file)
    return config


def prepare_model(config: ConfigDict):
    classifier = prepare_mlp_classifier(input_dim=config.model.input_dim, hidden_dims=config["model"]["hiddens"])
    model = Net(classifier)
    return model


def prepare_optimizer(model, lr_model):
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr_model)
    return optimizer


def prepare_criterion(pos_weight: float = None):
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        return nn.BCEWithLogitsLoss()


def run_evaluation(experiment_dir: str):
    config = prepare_config(experiment_dir)
    model = prepare_model(config)
    log_dir = os.path.join(experiment_dir, 'logs')
    assert os.path.exists(log_dir), f"No available experiments in {dir}"
    for experiment_name in os.listdir(log_dir):
        path_to_checkpoint = os.path.join(log_dir, experiment_name, "checkpoints", "best.pth")
        checkpoint = utils.load_checkpoint(path=path_to_checkpoint)
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=model,
        )
        experiment = ExperimentFactory(config)
        save_to = os.path.join(experiment_dir, "results")
        with open_log(save_to, name=experiment_name, mode=LoggingMode.OVERWRITE) as logdir:
            evaluate_model(model=model, loader=experiment.get_test_loader(), logdir=logdir)


def run_meta_experiment(experiment_dir: str, use_kde: bool = False, logging_mode: LoggingMode = LoggingMode.OVERWRITE):
    config = prepare_config(experiment_dir)
    model = prepare_model(config)
    optimizer = prepare_optimizer(model, config.hparams.lr_model)
    criterion = prepare_criterion()
    experiment = ExperimentFactory(config)
    utils = experiment.prepare_meta_experiment_with_smote()
    set_global_seed(config["seed"])
    runner = MetaClassificationRunner(dataset=utils["train_dataset"], use_kde=use_kde)
    with open_log(f'{experiment_dir}/logs', name=utils["name"], mode=logging_mode) as logdir:
        runner.train(model=model,
                     criterion=criterion,
                     optimizer=optimizer,
                     loaders=utils["loaders"],
                     logdir=logdir,
                     num_epochs=config["num_epochs"],
                     hparams=config["hparams"],
                     valid_loader="valid",
                     valid_metric="ap",
                     verbose=False,
                     minimize_valid_metric=False,
                     callbacks=OrderedDict({
                         "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(os.path.join(logdir, 'pr')),
                                                      loaders='valid'),
                         "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                                            input_key="scores",
                                                                                            target_key="targets"),
                                                      loaders='valid'
                                                      ),
                         "z": SaveSyntheticData(log_dir=logdir, save_best=True)
                     })
                     )
        # metrics = runner.log_evaluation_results(model=model,
        #                                         loader=utils["loaders"]["valid"],
        #                                         load_best=True, logger=logger)

    # return metrics


def run_baseline_experiment(experiment_dir: str, baseline: str = "base", logging_mode=LoggingMode.OVERWRITE):
    config = prepare_config(experiment_dir)
    model = prepare_model(config)
    optimizer = prepare_optimizer(model, config.hparams.lr_model)
    criterion = prepare_criterion()
    set_global_seed(config["seed"])
    experiment = ExperimentFactory(config)

    if baseline == "base":
        utils = experiment.prepare_base_experiment()
        model.classifier[-1].bias.data.fill_(np.log(1 / utils["ir"]))
    if baseline == "potential":
        utils = experiment.prepare_potential_experiment()
    if baseline == 'upsampling':
        utils = experiment.prepare_upsampling_experiment()
    if baseline == 'downsampling':
        utils = experiment.prepare_downsampling_experiment()
    if baseline == 'smote':
        utils = experiment.prepare_smote_experiment()

    with open_log(f'{experiment_dir}/logs', name=utils["name"], mode=logging_mode) as logdir:
        runner = ClassificationRunner()
        runner.train(model=model,
                     criterion=criterion,
                     optimizer=optimizer,
                     loaders=utils["loaders"],
                     logdir=logdir,
                     num_epochs=config["num_epochs"],
                     hparams=config["hparams"],
                     valid_loader="valid",
                     valid_metric="ap",
                     verbose=False,
                     minimize_valid_metric=False,
                     callbacks={
                         "periodic": dl.PeriodicLoaderCallback(valid_loader_key="valid", valid_metric_key="loss_x",
                                                               holdout=0),
                         "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(os.path.join(logdir, 'pr')),
                                                      loaders='valid'),
                         "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                                            input_key="scores",
                                                                                            target_key="targets"),
                                                      loaders='valid'
                                                      )
                     }
                     )

        # metrics = runner.log_evaluation_results(model=model,
        #                                         loader=utils["loaders"]["valid"],
        #                                         load_best=True, logger=logger)
    # return metrics


def run_keel_experiments():

    for data_name in os.listdir('./Keel1'):
        c_path = f'./Keel1/{data_name}'
        print("Processing ", c_path)
        with contextlib.redirect_stdout(io.StringIO()):
            # run_meta_experiment(c_path)
            run_evaluation(c_path)
            # run_baseline_experiment(c_path, baseline="smote")
        # run_meta_experiment(conf)
        # run_meta_experiment(c_path)


if __name__ == "__main__":
    # run_evaluation('./Keel1/glass2')
    run_keel_experiments()
    # conf = f'./Keel1/winequality-red-4/config.yml'
    # run_baseline_experiment(conf, baseline="base")
    # run_meta_experiment(conf, use_kde=False, logging_mode=LoggingMode.DEBUG)
