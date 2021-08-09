import torch
import numpy as np
import catalyst.dl as dl
import torch.nn.functional as F
from collections import OrderedDict
from models.networks import GaussianKDE
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.optim._functional import sgd
from catalyst import metrics
from datasets import SyntheticDataset
from typing import Mapping, Any, Optional, Dict
from experiment_utils import save_predictions
from models.net import Net


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, logdir: str):
    labels = []
    scores = []
    for batch in loader:
        x, y = batch['features'], batch['targets']
        y_hat = model(x)
        labels.append(y.numpy())
        scores.append(torch.sigmoid(y_hat).numpy())
    labels = np.concatenate(labels).squeeze()
    scores = np.concatenate(scores).squeeze()
    save_predictions(labels, scores, logdir)
    return labels, scores


@torch.no_grad()
def armijo_step(model: Net, z: torch.Tensor, z_grad: torch.Tensor,
                x: torch.Tensor, y: torch.Tensor,
                lr_meta: float, lr: float, k: int = 10):
    losses = torch.zeros(k, dtype=torch.float32)
    lr_arr = torch.zeros(k, dtype=torch.float32)
    for i in range(k):
        lr_i = lr / (2 ** i)
        test_z = z-lr_i*z_grad
        with torch.enable_grad():
            pz = model(test_z)
            loss_z = F.binary_cross_entropy_with_logits(pz, torch.ones_like(pz))
            gradients = grad(loss_z, model.parameters())
        logits = model(x, lr=lr_meta, gradients=gradients)
        loss_holdout = F.binary_cross_entropy_with_logits(logits, y.reshape_as(logits))
        losses[i] = loss_holdout
        lr_arr[i] = lr_i
    return lr_arr[torch.argmin(losses)]


class LoggingMixin:
    def log_figure(self, tag: str, figure: np.ndarray):
        tb_loggers = self.loggers['_tensorboard']
        tb_loggers._check_loader_key(loader_key=self.loader_key)
        writer = tb_loggers.loggers[self.loader_key]
        writer.add_figure(tag, figure, global_step=self.global_epoch_step)


class MetaClassificationRunner(dl.Runner, LoggingMixin):

    def __init__(self, dataset: SyntheticDataset, use_kde: bool = True, use_armijo=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.use_kde = use_kde
        self.use_armijo = use_armijo
        if use_kde:
            real = self.dataset.get_real_dataset()
            real_x, real_y = real["X"], real["y"]
            minority = torch.tensor(real_x[real_y == 1.], dtype=torch.float32)
            self.bw = torch.cdist(minority, minority).median()
            self.kde = GaussianKDE(X=minority, bw=self.bw)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        if self.loader_key == "train":
            loss_lst = ["loss", "loss_holdout"]
            if self.use_kde:
                loss_lst.append("loss_kde")
            self.meters = {
                key: metrics.AdditiveValueMetric(compute_on_call=False)
                for key in loss_lst
            }
        else:
            self.meters = {
                key: metrics.AdditiveValueMetric(compute_on_call=False)
                for key in ["loss"]
            }

    def on_loader_end(self, runner):
        for key in self.meters:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def on_epoch_end(self, runner: "IRunner"):
        runner.loaders["train"].dataset.shuffle_holdout_index()

    def get_datasets(self, stage: str) -> "OrderedDict[str, Dataset]":
        if stage == "train":
            return OrderedDict({"train": self.dataset})

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        if self.is_train_loader:
            is_synthetic = batch["is_synthetic"]
            index = batch["index"]
            x = batch["features"][~is_synthetic]
            y = batch["target"][~is_synthetic]
            synth_x = batch["features"][is_synthetic]
            synth_y = batch["target"][is_synthetic]
            holdout_x = batch["holdout_features"]
            holdout_y = batch["holdout_target"]

            # Calculate loss on the synthetic batch
            with torch.no_grad():
                logits = self.model(batch["features"])
                loss = F.binary_cross_entropy_with_logits(logits, batch["target"].reshape_as(logits))
                self.batch_metrics.update({"loss": loss})

            # Update model on real points
            px = self.model(x)
            loss = F.binary_cross_entropy_with_logits(px, y.reshape_as(px))
            gradients_x = grad(loss, self.model.parameters())

            # Calculate the gradient of the loss on the synthetic points
            z = synth_x.clone().detach().requires_grad_(True)
            pz = self.model(z)
            loss_z = F.binary_cross_entropy_with_logits(pz, synth_y.reshape_as(pz))
            gradients_z = grad(loss_z, self.model.parameters(), create_graph=True)

            # Take one gradient step with gradients obtained on z, and calculate loss on holdout
            gradients = tuple(gradients_z[i] + gradients_x[i] for i in range(len(gradients_z)))
            logits = self.model(holdout_x, lr=self.hparams["lr_meta"], gradients=gradients)
            loss_holdout = F.binary_cross_entropy_with_logits(logits, holdout_y.reshape_as(logits))
            loss_holdout.backward()
            self.batch_metrics.update({
                "loss_holdout": loss_holdout.data
            })

            if self.use_kde:
                z1 = torch.clone(z.detach())
                z1.requires_grad = True
                loss_kde = -self.kde.log_prob(z1)
                self.batch_metrics.update({"loss_kde": loss_kde})
                loss_kde.backward()
                z = z - self.hparams["lr_kde"] * z1.grad

            with torch.no_grad():
                sgd_params = dict(momentum_buffer_list=[],
                                  weight_decay=0.,
                                  momentum=0.,
                                  dampening=0.,
                                  nesterov=False)
                if self.use_armijo:
                    lr = armijo_step(self.model, z, z_grad=z.grad, x=holdout_x, y=holdout_y,
                                     lr_meta=self.hparams['lr_meta'],
                                     lr=self.hparams['lr_z']
                                     )
                else:
                    lr = self.hparams["lr_z"]
                sgd([z], z.grad, lr=lr, **sgd_params)
                z.grad.zero_()

            # Save the update for model with the gradients obtained on z
            self.model.gradient_step_(lr=self.hparams["lr_meta"], gradients=gradients)

            # Update
            ids = index[is_synthetic]
            self.loader.dataset.features[ids] = z.detach()

        elif self.is_valid_loader:
            x, y = batch['features'], batch['targets']
            y_hat = self.model(x)
            loss_x = F.binary_cross_entropy_with_logits(y_hat, y.reshape_as(y_hat))
            self.batch_metrics.update({
                "loss": loss_x
            })

            self.batch = {
                "features": x,
                "targets": y.view(-1, 1),
                "logits": y_hat,
                "scores": torch.sigmoid(y_hat.view(-1)),
            }

        for key in self.meters:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)


class ClassificationRunner(dl.Runner, LoggingMixin):

    def on_loader_start(self, runner: "IRunner"):
        super().on_loader_start(runner)
        self.meters = {
            "loss_x": metrics.AdditiveValueMetric(compute_on_call=False)
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

        self.batch_metrics.update({"loss_x": loss})
        self.meters["loss_x"].update(self.batch_metrics["loss_x"].item(), self.batch_size)

        self.batch = {
            "features": x,
            "targets": y.view(-1, 1),
            "logits": y_hat,
            "scores": torch.sigmoid(y_hat.view(-1)),
        }

    def on_loader_end(self, runner: "IRunner"):
        self.loader_metrics["loss_x"] = self.meters["loss_x"].compute()[0]
        super().on_loader_end(runner)
