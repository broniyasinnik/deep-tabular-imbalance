import torch
from catalyst import dl


class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device), batch[1].to(self.device))

    def _handle_batch(self, batch):
        x_cat, x_cont, y = batch
        y_hat = self.model(x_cat.type(torch.long), x_cont).squeeze()
        self.state.input = {"features_cat": x_cat, "features_cont": x_cont, "targets": y}
        self.state.output = {"logits": y_hat}
