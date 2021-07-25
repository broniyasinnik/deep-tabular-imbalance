import os
from collections import OrderedDict
from models.metrics import APMetric
from catalyst.utils.misc import set_global_seed
from callabacks import LogPRCurve, SaveSyntheticData
from catalyst import dl
from runners import ClassificationRunner, MetaClassificationRunner
from experiment_utils import experiment_logger, LoggingMode
from models.net import Net
from experiment_utils import ExperimentFactory, load_config, prepare_mlp_classifier


def run_meta_experiment(config_file: str, use_kde: bool = False, logging_mode: LoggingMode = LoggingMode.OVERWRITE):
    config = load_config(config_file)
    set_global_seed(config["seed"])
    classifier = prepare_mlp_classifier(input_dim=config.model.input_dim, hidden_dims=config["model"]["hiddens"])
    model = Net(classifier)
    experiment = ExperimentFactory(train_file=config["train_file"],
                                   test_file=config["test_file"],
                                   smote_file=config["smote_file"],
                                   model=model, hparams=config["hparams"])
    utils = experiment.prepare_meta_experiment_with_smote()
    runner = MetaClassificationRunner(dataset=utils["train_dataset"], use_kde=use_kde)
    with experiment_logger(f'{os.path.dirname(config_file)}/logs', name=utils["name"], mode=logging_mode) as logger:
        runner.train(model=model,
                     criterion=utils["criterion"],
                     optimizer=utils["optimizer"],
                     loaders=utils["loaders"],
                     logdir=logger.logdir,
                     num_epochs=config["num_epochs"],
                     hparams=config["hparams"],
                     valid_loader="valid",
                     valid_metric="ap",
                     verbose=False,
                     minimize_valid_metric=False,
                     callbacks=OrderedDict({
                         "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(os.path.join(logger.logdir, 'pr')),
                                                      loaders='valid'),
                         "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                                            input_key="scores",
                                                                                            target_key="targets"),
                                                      loaders='valid'
                                                      ),
                         "z": SaveSyntheticData(log_dir=logger.logdir, save_best=True)
                     })
                     )
        metrics = runner.log_evaluation_results(model=model,
                                                loader=utils["loaders"]["valid"],
                                                load_best=True, logger=logger)

    return metrics


def run_baseline_experiment(config_file: str, baseline: str = "base", logging_mode=LoggingMode.OVERWRITE):
    config = load_config(config_file)
    set_global_seed(config["seed"])

    classifier = prepare_mlp_classifier(input_dim=config.model.input_dim, hidden_dims=config["model"]["hiddens"])
    model = Net(classifier)

    experiment = ExperimentFactory(train_file=config["train_file"],
                                   test_file=config["test_file"],
                                   # holdout_file=config["holdout_file"],
                                   smote_file=config["smote_file"],
                                   model=model, hparams=config["hparams"])

    if baseline == "base":
        utils = experiment.prepare_base_experiment()
    if baseline == "potential":
        utils = experiment.prepare_potential_experiment()
    if baseline == 'upsampling':
        utils = experiment.prepare_upsampling_experiment()
    if baseline == 'downsampling':
        utils = experiment.prepare_downsampling_experiment()
    if baseline == 'smote':
        utils = experiment.prepare_smote_experiment()

    with experiment_logger(f'{os.path.dirname(config_file)}/logs', name=utils["name"], mode=logging_mode) as logger:
        runner = ClassificationRunner()
        runner.train(model=model,
                     criterion=utils["criterion"],
                     optimizer=utils["optimizer"],
                     loaders=utils["loaders"],
                     logdir=logger.logdir,
                     num_epochs=config["num_epochs"],
                     hparams=config["hparams"],
                     valid_loader="valid",
                     valid_metric="ap",
                     verbose=False,
                     minimize_valid_metric=False,
                     callbacks={
                         "periodic": dl.PeriodicLoaderCallback(valid_loader_key="valid", valid_metric_key="loss_x",
                                                               holdout=0),
                         "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(os.path.join(logger.logdir, 'pr')),
                                                      loaders='valid'),
                         "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                                            input_key="scores",
                                                                                            target_key="targets"),
                                                      loaders='valid'
                                                      )
                     }
                     )

        metrics = runner.log_evaluation_results(model=model,
                                                loader=utils["loaders"]["valid"],
                                                load_best=True, logger=logger)
    return metrics


def run_keel1_experiments():
    for data_name in os.listdir('./Keel1'):
        c_path = f'./Keel1/{data_name}/config.yml'
        print("Processing ", c_path)
        # run_baseline_experiment(c_path, baseline="upsampling")
        # run_meta_experiment(conf)
        run_meta_experiment(c_path)


if __name__ == "__main__":
    conf = f'./Keel1/winequality-red-4/config.yml'
    # run_baseline_experiment(conf, baseline="base")
    run_meta_experiment(conf, use_kde=False, logging_mode=LoggingMode.DEBUG)
