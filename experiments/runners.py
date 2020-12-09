import torch
import catalyst.dl as dl
from catalyst.core import Callback, CallbackOrder


class LogPRCurve(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.External)

    def on_epoch_end(self, runner):
        if runner.is_valid_loader:
            tb_callback = runner.callbacks["_tensorboard"]
            logger = tb_callback.loggers[runner.loader_name]
            model = runner.model
            valid_loader = runner.loaders[runner.loader_name]
            preds = []
            targs = []
            for batch, target in valid_loader:
                preds.append(torch.sigmoid(model(batch)))
                targs.append(target)
            tensorboard_probs = torch.cat(preds).squeeze()
            tensorboard_labels = torch.cat(targs)
            logger.add_pr_curve("precision_recall", tensorboard_labels, tensorboard_probs,
                                global_step=runner.epoch)


class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        x, y = batch
        return torch.sigmoid(self.model(x.to(self.device)))

    def _handle_batch(self, batch):
        x, y = batch
        y_hat = self.model(x).squeeze()
        self.state.input = {"features": x, "targets": y}
        self.state.output = {"logits": y_hat}
