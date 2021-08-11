import os
import io
import numpy as np
import contextlib
import optuna
from absl import logging
from absl import flags
from absl import app
from catalyst import utils
from visualization_utils import visualize_projection
from collections import OrderedDict
from models.metrics import APMetric
from catalyst.utils.misc import set_global_seed
from callabacks import LogPRCurve, SaveSyntheticData
from catalyst import dl
from runners import ClassificationRunner, MetaClassificationRunner, evaluate_model
from experiment_utils import open_log, LoggingMode
from experiment_utils import ExperimentFactory
from experiment_utils import get_config, get_model, get_test_loader, aggregate_results

FLAGS = flags.FLAGS
flags.DEFINE_string('exper_dir', './Adult/ir100/', help="Directory with experiment configuration")
flags.DEFINE_string('runs_dir', 'logs', help="Name of directory to save run logs")
flags.DEFINE_string('result_dir', 'results', help="Name of directory to save the results")
flags.DEFINE_string('visualization_dir', 'visualization', help="Name of directory to save visualization results")


class ExperimentRunner:
    def __init__(self, experiment_dir: str, runs_folder: str = "logs",
                 results_folder: str = "results", visualization_folder: str = "visualization",
                 trial: optuna.Trial = None):
        self.experiment_dir = experiment_dir
        self.runs_folder = runs_folder
        self.results_folder = results_folder
        self.visualization_folder = visualization_folder
        self.config = get_config(experiment_dir)
        self.trial = trial
        self.experiment_factory = ExperimentFactory(self.config)
        if self.trial is not None:
            self.runs_folder = 'optuna'

    def _get_callbacks(self, logdir, scheduler=None):
        callabacks = OrderedDict({
            "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(os.path.join(logdir, 'pr')),
                                         loaders='valid'),
            "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                               input_key="scores",
                                                                               target_key="targets"),
                                         loaders='valid'
                                         ),
            "auc": dl.ControlFlowCallback(
                base_callback=dl.AUCCallback(input_key="logits", target_key="targets"),
                loaders='valid'),

            "earlystopping": dl.EarlyStoppingCallback(patience=5, loader_key='valid', metric_key='ap',
                                                      minimize=False)
        })
        if scheduler is not None:
            callabacks["scheduler"] = dl.SchedulerCallback(loader_key='valid', metric_key='ap')

        if self.trial is not None:
            callabacks["optuna"] = dl.OptunaPruningCallback(
                loader_key="valid", metric_key="ap", minimize=False, trial=self.trial
            )
        return callabacks

    def run_visualization(self):
        if 'visualization' not in self.config:
            logging.info("Visualization configuration not found")
            return
        vis_conf = self.config.visualization
        save_to = os.path.join(FLAGS.exper_dir, FLAGS.visualization_dir)
        os.makedirs(save_to, exist_ok=True)
        for proj in vis_conf:
            visualize_projection(name=proj, save_to=save_to, **vis_conf[proj])

    def run_evaluation(self):

        runs_dir = os.path.join(self.experiment_dir, self.runs_folder)
        assert os.path.exists(runs_dir), f"No available experiments in {runs_dir}"

        # Default run evaluation on all experiment runs
        experiments = os.listdir(runs_dir)

        # Run evaluation on specified targets in config
        if 'evaluation' in self.config:
            experiments = self.config.evaluation.targets

        results = dict()
        save_to = os.path.join(self.experiment_dir, self.results_folder)
        model = get_model(self.config.model)
        test_loader = get_test_loader(self.config.test_file)

        for experiment_name in experiments:
            path_to_checkpoint = os.path.join(runs_dir, experiment_name, "checkpoints", "best.pth")
            checkpoint = utils.load_checkpoint(path=path_to_checkpoint)
            utils.unpack_checkpoint(
                checkpoint=checkpoint,
                model=model,
            )
            with open_log(save_to, name=experiment_name, mode=LoggingMode.OVERWRITE) as logdir:
                labels, scores = evaluate_model(model=model, loader=test_loader, logdir=logdir)
                results[experiment_name] = {
                    'labels': labels,
                    'scores': scores
                }

        with open_log(save_to, name='all', mode=LoggingMode.OVERWRITE) as logdir:
            aggregate_results(results, metrics=self.config.evaluation.metrics, logdir=logdir)

    def run_meta_experiment(self, name: str = 'meta', logging_mode: LoggingMode = LoggingMode.OVERWRITE):
        experiment = self.experiment_factory.prepare_meta_experiment_with_smote(name=name)
        set_global_seed(self.config["seed"])
        synth_data = experiment.loaders["train"].dataset
        runner = MetaClassificationRunner(dataset=synth_data, use_kde=experiment.hparams.use_kde,
                                          use_armijo=experiment.hparams.use_armijo)
        log_to = os.path.join(self.experiment_dir, self.runs_folder)
        with open_log(log_to, name=experiment.name, mode=logging_mode) as logdir:
            callbacks = self._get_callbacks(logdir)
            callbacks["save_synthetic"] = SaveSyntheticData(log_dir=logdir, save_best=True)
            runner.train(model=experiment.model,
                         loaders=experiment.loaders,
                         logdir=logdir,
                         num_epochs=experiment.epochs,
                         hparams=experiment.hparams,
                         valid_loader="valid",
                         valid_metric="ap",
                         verbose=False,
                         minimize_valid_metric=False,
                         callbacks=callbacks
                         )

    def run_baseline_experiment(self, name: str = "base", logging_mode=LoggingMode.OVERWRITE):
        set_global_seed(self.config["seed"])

        experiment = self.experiment_factory.prepare_baseline_experiment(name)
        if name == "base":
            experiment.model.classifier[-1].bias.data.fill_(np.log(1 / experiment.ir))

        log_to = os.path.join(self.experiment_dir, self.runs_folder)
        with open_log(log_to, name=experiment.name, mode=logging_mode) as logdir:
            runner = ClassificationRunner()
            callbacks = self._get_callbacks(logdir, scheduler=experiment.scheduler)
            runner.train(model=experiment.model,
                         criterion=experiment.criterion,
                         optimizer=experiment.optimizer,
                         scheduler=experiment.scheduler,
                         loaders=experiment.loaders,
                         logdir=logdir,
                         num_epochs=experiment.epochs,
                         hparams=self.config.experiments[name].to_dict(),
                         valid_loader="valid",
                         valid_metric="ap",
                         verbose=False,
                         minimize_valid_metric=False,
                         callbacks=callbacks
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


def main(argv):
    exper_dir = f'./Adult/ir200/'
    exper_runner = ExperimentRunner(exper_dir)
    # exper_runner.run_baseline_experiment(name='potential')
    # exper_runner.run_meta_experiment(name="meta")
    exper_runner.run_evaluation()
    return 0


if __name__ == "__main__":
    app.run(main)
