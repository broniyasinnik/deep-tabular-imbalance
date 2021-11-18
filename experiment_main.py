import os
from collections import OrderedDict
from typing import List

import optuna
from absl import app, flags, logging
from catalyst import dl, utils
from catalyst.utils.misc import set_global_seed

from callabacks import LogPRCurve, SaveSyntheticData
from experiment_utils import (
    ExperimentFactory,
    LoggingMode,
    get_config,
    open_log,
)
from models.metrics import APMetric

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "./experiments/adult_exper2/ir50/config.yml", help="Path to config file")
flags.DEFINE_list("targets", [], "Targets to run in experiment")
flags.DEFINE_string("logs_dir", "./experiments/adult_exper2/ir50/", help="Name of directory to save run logs")
flags.DEFINE_string("results_dir", "./experiments/adult_exper2/ir50/results",
                    help="Name of directory to save the results")
flags.DEFINE_string(
    "visualization_dir", "./experiments/adult_exper2/ir50/visualization",
    help="Name of directory to save visualization results"
)


class ExperimentRunner:
    def __init__(
            self,
            config_file: str,
            targets: List[str],
            logs_folder: str,
            results_folder: str,
            visualization_folder: str,
            trial: optuna.Trial = None,
    ):
        self.targets = targets
        self.logs_folder = logs_folder
        self.results_folder = results_folder
        self.visualization_folder = visualization_folder
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

    def run_experiments(self, logging_mode=LoggingMode.OVERWRITE):
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
                    hparams=self.config.experiments[experiment_name].get("hparams"),
                    valid_loader="valid",
                    valid_metric="ap",
                    verbose=False,
                    minimize_valid_metric=False,
                    callbacks=callbacks,
                )


# def run_keel_experiments():
#     for data_name in os.listdir("./Keel1"):
#         c_path = f"./Keel1/{data_name}"
#         print("Processing ", c_path)
#         runner = ExperimentRunner(c_path)
#         with contextlib.redirect_stdout(io.StringIO()):
#             # run_meta_experiment(c_path)
#             runner.run_evaluation(c_path)
#             # run_baseline_experiment(c_path, baseline="smote")
#         # run_meta_experiment(conf)
#         # run_meta_experiment(c_path)


def main(argv):
    runner = ExperimentRunner(FLAGS.config,
                              FLAGS.targets,
                              FLAGS.logs_dir,
                              FLAGS.results_dir,
                              FLAGS.visualization_dir)
    if len(argv) == 1:
        runner.run_experiments()
    elif argv[1] == "evaluate":
        runner.run_evaluation()
    return 0


if __name__ == "__main__":
    app.run(main)
