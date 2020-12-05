import torch
import catalyst.dl as dl


class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device), batch[1].to(self.device))

    def _handle_batch(self, batch):
        x, y = batch
        y_hat = self.model(x).squeeze()
        self.state.input = {"features": x, "targets": y}
        self.state.output = {"logits": y_hat}
