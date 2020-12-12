import torch
import catalyst.dl as dl
from torch.utils.tensorboard import SummaryWriter
from catalyst.core import Callback, CallbackOrder


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
        batch, targets = runner.input["features"], runner.input["targets"]
        probs = torch.sigmoid(runner.output["logits"])
        self.tensorboard_probs.append(probs)
        self.tensorboard_labels.append(targets)

    def on_stage_end(self, runner):
        if runner.is_infer_stage:
            tensorboard_probs = torch.cat(self.tensorboard_probs).squeeze()
            tensorboard_labels = torch.cat(self.tensorboard_labels).squeeze()
            self.writer.add_pr_curve("precision_recall", tensorboard_labels, tensorboard_probs)


class CustomRunner(dl.SupervisedRunner):

    # def predict_batch(self, batch):
    #     x, y = batch["features"], batch["targets"]
    #     return torch.sigmoid(self.model(x.to(self.device)))

    def _handle_batch(self, batch):
        x, y = batch["features"], batch["targets"]
        y_hat = self.model(x)
        preds = torch.sigmoid(self.model(x.to(self.device)))
        preds = torch.where(preds > 0.5, torch.ones_like(preds), torch.zeros_like(preds))
        self.state.input = {"features": x, "targets": y.view(-1, 1)}
        self.state.output = {"logits": y_hat, "preds": preds}
