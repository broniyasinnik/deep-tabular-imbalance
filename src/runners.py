from collections import OrderedDict
from copy import deepcopy
from typing import Any, Mapping
from itertools import cycle

import catalyst.dl as dl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from catalyst import metrics
from catalyst.typing import Dataset
from torch.autograd import grad

from models.net import Net


@torch.no_grad()
def armijo_step_z(
        model: Net,
        z: torch.Tensor,
        z_grad: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        lr_meta: float,
        lr: float,
        k: int = 10,
):
    losses = torch.zeros(k, dtype=torch.float32)
    lr_arr = torch.zeros(k, dtype=torch.float32)
    for i in range(k):
        lr_i = lr / (2 ** i)
        test_z = z - lr_i * z_grad
        with torch.enable_grad():
            pz = model(test_z)
            loss_z = F.binary_cross_entropy_with_logits(pz, torch.ones_like(pz))
            gradients = grad(loss_z, model.parameters())
        logits = model(x, lr=lr_meta, gradients=gradients)
        loss_holdout = F.binary_cross_entropy_with_logits(logits, y.reshape_as(logits))
        losses[i] = loss_holdout
        lr_arr[i] = lr_i
    return lr_arr[torch.argmin(losses)]


@torch.no_grad()
def armijo_step_model(
        model: Net,
        gradients: torch.Tensor,
        steps: int,
        lr_meta: float,
        x: torch.Tensor,
        y: torch.Tensor,
):
    num_coords = len(gradients)
    lr = torch.tensor([lr_meta / (2 ** i) for i in range(steps)], dtype=torch.float32)
    lr_total = torch.zeros(num_coords)
    for i in range(num_coords):
        e_i = torch.eye(num_coords)[i, :]
        loss = 1e9 * torch.ones(steps)
        for j in range(steps):
            model_i = deepcopy(model)
            lr_i = e_i * lr[j]
            model_i.gradient_step_(lr_i, gradients)
            logits = model_i(x)
            loss[j] = F.binary_cross_entropy_with_logits(logits, y.reshape_as(logits))
        lr_total[i] = lr[torch.argmin(loss)]

    return lr_total


class LoggingMixin:
    def log_figure(self, tag: str, figure: np.ndarray):
        tb_loggers = self.loggers["_tensorboard"]
        tb_loggers._check_loader_key(loader_key=self.loader_key)
        writer = tb_loggers.loggers[self.loader_key]
        writer.add_figure(tag, figure, global_step=self.global_epoch_step)


class MetaClassificationRunner(dl.Runner, LoggingMixin):

    def __init__(self):
        super().__init__()
        self.meters_train = dict()
        self.meters_valid = dict()
        self.kde = None

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        if self.loader_key == "train":
            self.meters_train = {
                key: metrics.AdditiveMetric(compute_on_call=False) for key in ["loss", "loss_emulative"]
            }
        elif self.loader_key == "valid":
            self.meters_valid = {
                key: metrics.AdditiveMetric(compute_on_call=False) for key in ["loss"]
            }

    def on_loader_end(self, runner):
        if self.loader_key == "train":
            for key in self.meters_train:
                self.loader_metrics[key] = self.meters_train[key].compute()[0]
            if self.kde is not None:
                kde_score = self.kde.score(self.synthetic_loader.dataset.features)
                self.loader_metrics["kde"] = kde_score
        elif self.loader_key == "valid":
            for key in self.meters_valid:
                self.loader_metrics[key] = self.meters_valid[key].compute()[0]

        super().on_loader_end(runner)

    def set_kde_model(self, kde_model):
        self.kde = kde_model

    def set_training_loaders(self, synthetic_loader: DataLoader, train_hold_out_loader: DataLoader):
        self.synthetic_loader = synthetic_loader
        self.train_hold_out_loader = train_hold_out_loader
        self.synthetic_batch_iterator = cycle(synthetic_loader)
        self.train_hold_out_iterator = cycle(train_hold_out_loader)

    def get_datasets(self, stage: str) -> "OrderedDict[str, Dataset]":
        if stage == "train":
            return OrderedDict({"train": self.dataset})

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        if self.is_train_loader:
            x = batch["features"]
            yx = batch["targets"]
            batch_hold_out = next(self.train_hold_out_iterator)
            xh = batch_hold_out["features"]
            yh = batch_hold_out["targets"]
            batch_synthetic = next(self.synthetic_batch_iterator)
            z = batch_synthetic["features"]
            yz = batch_synthetic["targets"]
            items = batch_synthetic["index"]

            # Calculate the gradient of the loss on the synthetic points
            z = z.clone().detach().requires_grad_(True)
            optimizer = torch.optim.SGD([z], lr=self.hparams["lr_data"], momentum=0.0)
            pz = self.model(torch.cat([z, x]))
            loss = F.binary_cross_entropy_with_logits(pz, torch.cat([yz, yx]).reshape_as(pz))
            gradients_z = grad(loss, self.model.parameters(), create_graph=True)
            
            ph = self.model(xh, lr=self.hparams["lr_model"], gradients=gradients_z)
            # logits.clamp(min=-10, max=10)
            loss_emulative = F.binary_cross_entropy_with_logits(
                ph, yh.reshape_as(ph)
            )
            # log metrics
            self.batch_metrics.update({"loss": loss.data})
            self.batch_metrics.update({"loss_emulative": loss_emulative.data})
            self.meters_train["loss"].update(self.batch_metrics["loss"].item(),
                                             self.batch_size + self.hparams.batch_synthetic)
            self.meters_train["loss_emulative"].update(self.batch_metrics["loss_emulative"].item(),
                                                       self.hparams.batch_holdout)
            loss_emulative.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update model
            self.model.gradient_step_(
                lr=self.hparams["lr_model"], gradients=gradients_z, alpha=self.hparams["alpha"]
            )
            # Save updated z to dataset
            self.synthetic_loader.dataset.features[items] = z.detach()

        elif self.is_valid_loader:
            x, y = batch["features"], batch["targets"]
            y_hat = self.model(x)
            # y_hat.clamp(min=-10, max=10)
            loss_x = F.binary_cross_entropy_with_logits(y_hat, y.reshape_as(y_hat))
            self.batch_metrics.update({"loss": loss_x})
            self.meters_valid["loss"].update(self.batch_metrics["loss"].item(), self.batch_size)

            self.batch = {
                "features": x,
                "targets": y.view(-1, 1),
                "logits": y_hat,
                "scores": torch.sigmoid(y_hat.view(-1)),
            }


class ClassificationRunner(dl.Runner, LoggingMixin):
    def on_loader_start(self, runner: "IRunner"):
        super().on_loader_start(runner)
        self.meters = {"loss": metrics.AdditiveValueMetric(compute_on_call=False)}

    def handle_batch(self, batch):
        x, y = batch["features"], batch["targets"]
        y_hat = self.model(x)
        if self.is_train_loader:
            loss = self.criterion(y_hat, y.reshape_as(y_hat))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        elif self.is_valid_loader:
            loss = self.criterion(y_hat, y.reshape_as(y_hat))

        self.batch_metrics.update({"loss": loss})
        self.meters["loss"].update(self.batch_metrics["loss"].item(), self.batch_size)

        self.batch = {
            "features": x,
            "targets": y.view(-1, 1),
            "logits": y_hat,
            "scores": torch.sigmoid(y_hat.view(-1)),
        }

    def on_loader_end(self, runner: "IRunner"):
        self.loader_metrics["loss"] = self.meters["loss"].compute()[0]
        super().on_loader_end(runner)
