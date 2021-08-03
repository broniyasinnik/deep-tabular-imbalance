import os
import io
import torch
import numpy as np
import contextlib
import optuna
from catalyst import utils
from collections import OrderedDict
from models.metrics import APMetric
from catalyst.utils.misc import set_global_seed
from callabacks import LogPRCurve, SaveSyntheticData
from catalyst import dl
from runners import ClassificationRunner, MetaClassificationRunner, evaluate_model
from experiment_utils import open_log, LoggingMode
from experiment_utils import ExperimentFactory
from experiment_utils import prepare_config, prepare_model, prepare_criterion, prepare_optimizer, prepare_scheduler


class ExperimentRunner:
    def __init__(self, experiment_dir: str, trial: optuna.Trial = None):
        self.experiment_dir = experiment_dir
        self.config = prepare_config(experiment_dir)
        self.model = prepare_model(self.config)
        self.optimizer = prepare_optimizer(self.model, self.config.hparams.lr_model, self.hparams.weight_decay)
        self.criterion = prepare_criterion()
        self.scheduler = prepare_scheduler()

        self.trial = trial
        self.experiment_factory = ExperimentFactory(self.config)
        if self.trial is not None:
            self.log_to = f'{self.experiment_dir}/optuna_logs'
        else:
            self.log_to = f'{self.experiment_dir}/logs'

    def _get_callbacks(self, logdir):
        callabacks = OrderedDict({
            "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(os.path.join(logdir, 'pr')),
                                         loaders='valid'),
            "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                               input_key="scores",
                                                                               target_key="targets"),
                                         loaders='valid'
                                         ),
            "scheduler": dl.SchedulerCallback(loader_key='valid', metric_key='ap'),
            "auc": dl.ControlFlowCallback(
                base_callback=dl.AUCCallback(input_key="logits", target_key="targets"),
                loaders='valid')
        })
        if self.trial is not None:
            callabacks["optuna"] = dl.OptunaPruningCallback(
                loader_key="valid", metric_key="ap", minimize=False, trial=self.trial
            )
        return callabacks

    def run_evaluation(self):
        log_dir = os.path.join(self.experiment_dir, 'logs')
        assert os.path.exists(log_dir), f"No available experiments in {dir}"
        for experiment_name in os.listdir(log_dir):
            path_to_checkpoint = os.path.join(log_dir, experiment_name, "checkpoints", "best.pth")
            checkpoint = utils.load_checkpoint(path=path_to_checkpoint)
            utils.unpack_checkpoint(
                checkpoint=checkpoint,
                model=self.model,
            )
            experiment = ExperimentFactory(self.config)
            save_to = os.path.join(self.experiment_dir, "results")
            with open_log(save_to, name=experiment_name, mode=LoggingMode.OVERWRITE) as logdir:
                evaluate_model(model=self.model, loader=experiment.get_test_loader(), logdir=logdir)

    def run_meta_experiment(self, use_kde: bool = False, logging_mode: LoggingMode = LoggingMode.OVERWRITE):
        utils = self.experiment_factory.prepare_meta_experiment_with_smote()
        set_global_seed(self.config["seed"])
        runner = MetaClassificationRunner(dataset=utils["train_dataset"], use_kde=use_kde)

        with open_log(self.log_to, name=utils["name"], mode=logging_mode) as logdir:
            callbacks = self._get_callbacks(logdir)
            callbacks["save_synthetic"] = SaveSyntheticData(log_dir=logdir, save_best=True)
            runner.train(model=self.model,
                         criterion=self.criterion,
                         optimizer=self.optimizer,
                         loaders=utils["loaders"],
                         logdir=logdir,
                         num_epochs=self.config["num_epochs"],
                         hparams=self.config["hparams"],
                         valid_loader="valid",
                         valid_metric="ap",
                         verbose=False,
                         minimize_valid_metric=False,
                         callbacks=callbacks
                         )

    def run_baseline_experiment(self, baseline: str = "base", logging_mode=LoggingMode.OVERWRITE):
        set_global_seed(self.config["seed"])

        if baseline == "base":
            utils = self.experiment_factory.prepare_base_experiment()
            self.model.classifier[-1].bias.data.fill_(np.log(1 / utils["ir"]))
        if baseline == "potential":
            utils = self.experiment_factory.prepare_potential_experiment()
        if baseline == 'upsampling':
            utils = self.experiment_factory.prepare_upsampling_experiment()
        if baseline == 'downsampling':
            utils = self.experiment_factory.prepare_downsampling_experiment()
        if baseline == 'smote':
            utils = self.experiment_factory.prepare_smote_experiment()


        with open_log(self.log_to, name=utils["name"], mode=logging_mode) as logdir:
            runner = ClassificationRunner()
            runner.train(model=self.model,
                         criterion=self.criterion,
                         optimizer=self.optimizer,
                         loaders=utils["loaders"],
                         logdir=logdir,
                         num_epochs=self.config["num_epochs"],
                         hparams=self.config["hparams"],
                         valid_loader="valid",
                         valid_metric="ap",
                         verbose=False,
                         scheduler=self.scheduler,
                         minimize_valid_metric=False,
                         callbacks=self._get_callbacks(logdir)
                         )


def run_keel_experiments():
    for data_name in os.listdir('./Keel1'):
        c_path = f'./Keel1/{data_name}'
        print("Processing ", c_path)
        runner = ExperimentRunner(c_path)
        with contextlib.redirect_stdout(io.StringIO()):
            # run_meta_experiment(c_path)
            runner.run_evaluation(c_path)
            # run_baseline_experiment(c_path, baseline="smote")
        # run_meta_experiment(conf)
        # run_meta_experiment(c_path)


if __name__ == "__main__":
    # run_evaluation('./Keel1/glass2')
    # run_keel_experiments()
    exper_dir = f'./Adult/ir100/'
    exper_runner = ExperimentRunner(exper_dir)
    exper_runner.run_baseline_experiment("upsampling")
    # exper_runner.run_meta_experiment()
    # run_evaluation(dir)
    # run_meta_experiment(conf, use_kde=False, logging_mode=LoggingMode.DEBUG)
