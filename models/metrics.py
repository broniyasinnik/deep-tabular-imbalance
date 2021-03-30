import torch
import numpy as np
from typing import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from catalyst import dl, metrics, utils


class APMetric(metrics.ICallbackLoaderMetric):

    def __init__(self, compute_on_call: bool = True, prefix: str = None,
                 suffix: str =None):
        super(APMetric, self).__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}ap{self.suffix}"
        self.scores = []
        self.targets = []

    def reset(self, num_batches, num_samples) -> None:
        self.scores = []
        self.targets = []

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        self.scores.append(scores.cpu().detach())
        self.targets.append(targets.cpu().detach())

    def compute(self):
        scores = torch.cat(self.scores)
        targets = torch.cat(self.targets)
        ap = average_precision_score(targets, scores)
        return ap

    def compute_key_value(self) -> Dict[str, float]:
        ap = self.compute()
        return {self.metric_name: ap}


class BalancedAccuracyMetric(metrics.ICallbackBatchMetric, metrics.AdditiveValueMetric):
    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> float:
        scores = scores.detach().to('cpu').numpy()
        y_true = targets.detach().to('cpu').numpy()
        y_pred = np.where(scores >= 0.5, 1, 0)
        value = balanced_accuracy_score(y_true, y_pred)
        value = super().update(value, len(targets))
        return value

    def update_key_value(self, scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        value = self.update(scores, targets)
        return {"accuracy": value}

    def compute_key_value(self) -> Dict[str, float]:
        mean, std = super().compute()
        return {"accuracy": mean, "accuracy/std": std}


