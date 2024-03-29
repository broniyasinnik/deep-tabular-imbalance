import os
from collections import OrderedDict
from pathlib import Path
from typing import List

import optuna
from absl import app, flags, logging
from catalyst import dl, utils
from catalyst.utils.misc import set_global_seed
from torch.utils.tensorboard import SummaryWriter

from callabacks import LogPRCurve
from evaluation_utils import evaluate_metrics, model_predictions
from experiment_utils import (
    ExperimentFactory,
    LoggingMode,
    get_config,
    open_log,
)
from models.metrics import APMetric

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", "./experiments/adult_exper2/ir50/config.yml", help="Path to config file"
)
flags.DEFINE_list("targets", [], "Targets to run in experiment")
flags.DEFINE_string(
    "logs_dir", "./experiments/adult_exper2/ir50/", help="Name of directory to save run logs"
)
flags.DEFINE_string(
    "results_dir",
    "./experiments/adult_exper2/ir50/results",
    help="Name of directory to save the results",
)
flags.DEFINE_string(
    "visualization_dir",
    "./experiments/adult_exper2/ir50/visualization",
    help="Name of directory to save visualization results",
)


class ExperimentRunner:
    def __init__(
            self,
            config_file: str,
            targets: List[str],
            logs_folder: str,
            trial: optuna.Trial = None,
    ):
        self.targets = targets
        self.logs_folder = logs_folder
        self.config = get_config(config_file)
        self.trial = trial
        self.experiment_factory = ExperimentFactory(self.config)
        if self.trial is not None:
            self.runs_folder = "optuna"

    def _get_callbacks(self, logdir, scheduler=None):
        callabacks = OrderedDict(
            {
                "pr": dl.ControlFlowCallback(
                    base_callback=LogPRCurve(os.path.join(logdir, "pr")), loaders="valid"
                ),
                "ap": dl.ControlFlowCallback(
                    base_callback=dl.LoaderMetricCallback(
                        metric=APMetric(), input_key="scores", target_key="targets"
                    ),
                    loaders="valid",
                ),
                "auc": dl.ControlFlowCallback(
                    base_callback=dl.AUCCallback(input_key="logits", target_key="targets"),
                    loaders="valid",
                ),
                # "earlystopping": dl.EarlyStoppingCallback(
                #     patience=20, loader_key="valid", metric_key="ap", minimize=False
                # ),
            }
        )
        if scheduler is not None:
            callabacks["scheduler"] = dl.SchedulerCallback(loader_key="valid", metric_key="ap")

        if self.trial is not None:
            callabacks["optuna"] = dl.OptunaPruningCallback(
                loader_key="valid", metric_key="ap", minimize=False, trial=self.trial
            )
        return callabacks

    def run_experiments(self, logging_mode=LoggingMode.NORMAL):
        set_global_seed(self.config["seed"])

        for experiment_name in self.targets:
            logging.info(f"Starting experiment {experiment_name}...")
            experiment = self.experiment_factory.prepare_experiment(experiment_name)
            with open_log(self.logs_folder, name=experiment.name, mode=logging_mode) as logdir:
                runner = experiment.runner
                callbacks = self._get_callbacks(logdir, scheduler=experiment.scheduler)
                runner.train(
                    model=experiment.model,
                    criterion=experiment.criterion,
                    optimizer=experiment.optimizer,
                    scheduler=experiment.scheduler,
                    loaders=experiment.loaders,
                    logdir=logdir,
                    num_epochs=experiment.epochs,
                    hparams=experiment.hparams,
                    valid_loader="valid",
                    valid_metric="ap",
                    verbose=True,
                    minimize_valid_metric=False,
                    callbacks=callbacks,
                    load_best_on_end=True,
                )
                # Load best model
                model = experiment.model
                checkpoint = utils.load_checkpoint(path=Path(logdir)/"checkpoints"/"best.pth")
                utils.unpack_checkpoint(
                    checkpoint=checkpoint,
                    model=model,
                )
                # Save best model predictions
                save_to = Path(logdir) / "results"
                save_to.mkdir(parents=True, exist_ok=True)
                model_predictions(
                    model,
                    data_file=self.config.train_file,
                    save_to=save_to / "train_predictions.csv",
                )
                labels, scores = model_predictions(
                    model,
                    data_file=self.config.valid_file,
                    save_to=save_to / "valid_predictions.csv",
                )
                with SummaryWriter(log_dir=Path(logdir) / "tensorboard" / "hparms") as w:
                    metric_dict = evaluate_metrics(labels, scores)
                    w.add_hparams(hparam_dict=experiment.hparams.to_dict(), metric_dict=metric_dict)


def main(argv):
    runner = ExperimentRunner(
        FLAGS.config, FLAGS.targets, FLAGS.logs_dir
    )
    runner.run_experiments()
    return 0


if __name__ == "__main__":
    app.run(main)
