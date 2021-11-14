from collections import OrderedDict
from copy import deepcopy
from typing import Any, Mapping

import catalyst.dl as dl
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from catalyst import metrics
from catalyst.typing import Dataset
from torch.autograd import grad
from torch.optim._functional import sgd
from torch.utils.data import DataLoader

from models.net import Net
from models.networks import GaussianKDE



@torch.no_grad()
def armijo_step_z(model: Net, z: torch.Tensor, z_grad: torch.Tensor,
                  x: torch.Tensor, y: torch.Tensor,
                  lr_meta: float, lr: float, k: int = 10):
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
def armijo_step_model(model: Net, gradients: torch.Tensor, steps: int, lr_meta: float, x: torch.Tensor,
                      y: torch.Tensor):
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
        tb_loggers = self.loggers['_tensorboard']
        tb_loggers._check_loader_key(loader_key=self.loader_key)
        writer = tb_loggers.loggers[self.loader_key]
        writer.add_figure(tag, figure, global_step=self.global_epoch_step)


class MetaClassificationRunner(dl.Runner, LoggingMixin):

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss"]
        }

    def on_loader_end(self, runner):
        for key in self.meters:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def get_datasets(self, stage: str) -> "OrderedDict[str, Dataset]":
        if stage == "train":
            return OrderedDict({"train": self.dataset})

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        if self.is_train_loader:
            x = batch["features_x"]
            yx = batch["targets_x"]
            z = batch["features_z"]
            yz = batch["targets_z"]
            items = batch["item"]

            # Calculate the gradient of the loss on the synthetic points
            z = z.clone().detach().requires_grad_(True)
            optimizer = torch.optim.SGD([z], lr=self.hparams["lr_z"], momentum=0.0)
            pz = self.model(z)
            loss_z = F.binary_cross_entropy_with_logits(pz, yz.reshape_as(pz))
            gradients_z = grad(loss_z, self.model.parameters(), create_graph=True)

            logits = self.model(x, lr=self.hparams["lr_meta"], gradients=gradients_z)
            # logits.clamp(min=-10, max=10)
            loss_holdout = F.binary_cross_entropy_with_logits(logits, yx.reshape_as(logits))
            # log metrics
            self.batch_metrics.update({
                "loss": loss_holdout.data
            })
            self.meters["loss"].update(self.batch_metrics["loss"].item(), self.batch_size)
            loss_holdout.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update model
            self.model.gradient_step_(lr=self.hparams["lr_meta"], gradients=gradients_z, alpha=self.hparams["alpha"])
            # Save updated z to dataset
            self.loader.dataset.features_synthetic[items] = z.detach()

            # is_synthetic = batch["is_synthetic"]
            # index = batch["index"]
            # x = batch["features"][~is_synthetic]
            # y = batch["target"][~is_synthetic]
            # synth_x = batch["features"][is_synthetic]
            # synth_y = batch["target"][is_synthetic]
            # holdout_x = batch["holdout_features"]
            # holdout_y = batch["holdout_target"]

            # Calculate loss on the synthetic batch
            # with torch.no_grad():
            #     logits = self.model(batch["features"])
            #     logits.clamp(min=-10, max=10)
            #     loss = F.binary_cross_entropy_with_logits(logits, batch["target"].reshape_as(logits))
            #     self.batch_metrics.update({"loss": loss})

            # Calculate model on real points
            # px = self.model(x)
            # loss = F.binary_cross_entropy_with_logits(px, y.reshape_as(px))
            # gradients_x = grad(loss, self.model.parameters())

            # Calculate the gradient of the loss on the synthetic points
            # z = synth_x.clone().detach().requires_grad_(True)
            # pz = self.model(z)
            # loss_z = F.binary_cross_entropy_with_logits(pz, synth_y.reshape_as(pz))
            # gradients_z = grad(loss_z, self.model.parameters(), create_graph=True)

            # Take one gradient step with gradients obtained on z, and calculate loss on holdout
            # gradients = tuple(gradients_z[i] + gradients_x[i] for i in range(len(gradients_z)))
            # logits = self.model(holdout_x, lr=self.hparams["lr_meta"], gradients=gradients)
            # logits.clamp(min=-10, max=10)
            # loss_holdout = F.binary_cross_entropy_with_logits(logits, holdout_y.reshape_as(logits))
            # loss_holdout.backward()
            # self.batch_metrics.update({
            #     "loss_holdout": loss_holdout.data
            # })
            #
            # if self.use_kde:
            #     z1 = torch.clone(z.detach())
            #     z1.requires_grad = True
            #     loss_kde = -self.kde.log_prob(z1)
            #     self.batch_metrics.update({"loss_kde": loss_kde})
            #     loss_kde.backward()
            #     z = z - self.hparams["lr_kde"] * z1.grad
            #
            # with torch.no_grad():
            #     sgd_params = dict(momentum_buffer_list=[],
            #                       weight_decay=0.,
            #                       momentum=0.,
            #                       dampening=0.,
            #                       nesterov=False)
            #     if self.use_armijo:
            #         lr = armijo_step_model(self.model, param_grad=gradients,
            #                                lr_meta=self.hparams['lr_meta'])
            #         self.batch_metrics.update(lr_z=float(lr))
            #     else:
            #         lr = self.hparams["lr_z"]
            #     sgd([z], z.grad, lr=lr, **sgd_params)
            #     z.grad.zero_()
            #
            # # Save the update for model with the gradients obtained on z
            # self.model.gradient_step_(lr=self.hparams["lr_meta"], gradients=gradients)
            #
            # # Update
            # ids = index[is_synthetic]
            # self.loader.dataset.features[ids] = z.detach()

        elif self.is_valid_loader:
            x, y = batch['features'], batch['targets']
            y_hat = self.model(x)
            # y_hat.clamp(min=-10, max=10)
            loss_x = F.binary_cross_entropy_with_logits(y_hat, y.reshape_as(y_hat))
            self.batch_metrics.update({
                "loss": loss_x
            })
            self.meters["loss"].update(self.batch_metrics["loss"].item(), self.batch_size)

            self.batch = {
                "features": x,
                "targets": y.view(-1, 1),
                "logits": y_hat,
                "scores": torch.sigmoid(y_hat.view(-1)),
            }



class ClassificationRunner(dl.Runner, LoggingMixin):

    def on_loader_start(self, runner: "IRunner"):
        super().on_loader_start(runner)
        self.meters = {
            "loss": metrics.AdditiveValueMetric(compute_on_call=False)
        }

    def handle_batch(self, batch):
        x, y = batch['features'], batch['targets']
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
