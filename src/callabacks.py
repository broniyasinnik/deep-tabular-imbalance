import os
import torch
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from catalyst.core import Callback, CallbackOrder
from sklearn.metrics import balanced_accuracy_score
from visualization_utils import visualize_decision_boundary


class BalancedAccuracyCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Metric)

    @torch.no_grad()
    def on_batch_end(self, runner: "IRunner") -> None:
        y_true = runner.batch["targets"].to("cpu").numpy()
        scores = runner.batch["scores"].to("cpu").numpy()
        y_pred = np.where(scores >= 0.5, 1, 0)
        runner.batch_metrics["balanced_accuracy_score"] = balanced_accuracy_score(y_true, y_pred)


class LogPRCurve(Callback):
    def __init__(self, log_dir: str = None):
        super().__init__(CallbackOrder.External)
        self.tensorboard_probs = []
        self.tensorboard_labels = []
        self.writer = SummaryWriter(log_dir)

    def on_loader_start(self, runner):
        self.tensorboard_probs = []
        self.tensorboard_labels = []

    def on_batch_end(self, runner):
        targets = runner.batch["targets"]
        probs = runner.batch["scores"]
        self.tensorboard_probs.append(probs)
        self.tensorboard_labels.append(targets)

    def on_loader_end(self, runner):
        tensorboard_probs = torch.cat(self.tensorboard_probs).squeeze()
        tensorboard_labels = torch.cat(self.tensorboard_labels).squeeze()
        self.writer.add_pr_curve("precision_recall", tensorboard_labels, tensorboard_probs,
                                 global_step=runner.stage_epoch_step)


class SaveSyntheticData(Callback):
    def __init__(self, log_dir: str, save_best: bool = True, save_last: bool = False):
        super().__init__(order=CallbackOrder.External)
        self.log_dir = os.path.join(log_dir, "z_synthetic")
        self.save_best = save_best
        self.save_last = save_last
        self.best_valid_metric = None

    def is_better(self, epoch_metric, minimize: bool):
        if minimize and epoch_metric <= self.best_valid_metric:
            return True
        if not minimize and epoch_metric > self.best_valid_metric:
            return True
        return False

    def on_epoch_end(self, runner: "IRunner") -> None:
        if self.save_best:
            dataset = runner.get_datasets(stage=runner.stage_key)["train"]
            synthetic = dataset.synthetic_dataset
            epoch_metric = runner.epoch_metrics["valid"][runner._valid_metric]
            if self.best_valid_metric is None:
                self.best_valid_metric = epoch_metric
            if self.is_better(epoch_metric, minimize=runner._minimize_valid_metric):
                np.savez(self.log_dir, X=synthetic["X"], y=synthetic["y"])
                self.best_valid_metric = epoch_metric
        if self.save_last:
            # @TODO: Add option for saving the last
            ...


class DecisionBoundaryCallback(Callback):
    def __init__(self, show=False):
        super().__init__(order=CallbackOrder.External)
        self.show = show

    def on_loader_end(self, runner):
        loader = runner.loader
        image_boundary = visualize_decision_boundary(loader.dataset, runner.model)
        if self.show:
            display.clear_output(wait=True)
            plt.show()
        runner.log_figure(tag=f"decision_boundary_{runner.loader_key}", figure=image_boundary)
