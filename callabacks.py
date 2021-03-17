import torch
from IPython import display
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from catalyst.core import Callback, CallbackOrder
from sklearn.metrics import balanced_accuracy_score
from visualization_utils import visualize_decision_boundary, visualize_dataset
import matplotlib.pyplot as plt
import time
import numpy as np


class BalancedAccuracyCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, runner: "IRunner") -> None:
        y_true = runner.input["targets"].to("cpu").numpy()
        y_pred = runner.output["preds"].to("cpu").numpy()
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
        targets = runner.input["targets"]
        probs = torch.sigmoid(runner.output["logits"])
        self.tensorboard_probs.append(probs)
        self.tensorboard_labels.append(targets)

    def on_stage_end(self, runner):
        if runner.is_infer_stage:
            tensorboard_probs = torch.cat(self.tensorboard_probs).squeeze()
            tensorboard_labels = torch.cat(self.tensorboard_labels).squeeze()
            self.writer.add_pr_curve("precision_recall", tensorboard_labels, tensorboard_probs)


class DecisionBoundaryCallback(Callback):
    def __init__(self, plot_synthetic=False):
        super().__init__(order=CallbackOrder.External)
        self.plot_synthetic = plot_synthetic


    def on_epoch_end(self, runner):
        loader = runner.loaders["valid"]
        if type(loader.dataset) is TensorDataset:
            X = loader.datset.tensors[0]
            y = loader.dataset.tensors[1]
        else:
            X = loader.dataset.data
            y = loader.dataset.target
        image_boundry = visualize_decision_boundary(X, y, runner.model, self.plot_synthetic)
        display.clear_output(wait=True)
        plt.show()
        runner.log_figure(tag="decision_boundary", figure=image_boundry)


class HypernetVisualization(Callback):
    def __init__(self):
        super().__init__(order=CallbackOrder.External)

    def on_epoch_end(self, runner):
        X_data = runner.loaders["train"].dataset.data
        y_data = runner.loaders["train"].dataset.target
        X_syn = runner.model["hypernetwork"].x_syn.detach().numpy()
        y_syn = runner.model["hypernetwork"].y_syn.squeeze()
        X = np.concatenate([X_data, X_syn])
        y = np.concatenate([y_data, y_syn])
        image_dataset = visualize_dataset(X, y)
        runner.log_figure(tag="syntetic", figure=image_dataset)

class LogGANProjection(Callback):
    def __init__(self, log_dir: str, tag: str, samples: int = 200):
        super().__init__(CallbackOrder.External)
        self.writer = SummaryWriter(log_dir)
        self.samples = samples
        self.tag = tag

    def on_stage_end(self, runner: "IRunner") -> None:
        embbedings_dict = runner.log_embeddings()
        embeddings = []
        metadata = []
        # Collect embeddings from loaders
        for key in embbedings_dict:
            loader_embbedings = embbedings_dict[key]["embedding"]
            embeddings.append(loader_embbedings)
            loader_labels = embbedings_dict[key]["labels"]
            labels = [f'{key}:{int(label)}' for label in loader_labels]
            metadata += labels

        # Collect embeddings from GAN
        random_latent_vectors, generated_labels = runner.model["generator"].sample_latent_vectors(self.samples)
        generated_embeddings = runner.model["generator"](random_latent_vectors).to('cpu')
        embeddings.append(generated_embeddings)
        gan_labels = [f'gan:{int(label)}' for label in generated_labels]
        metadata += gan_labels

        embeddings = torch.cat(embeddings)
        self.writer.add_embedding(embeddings, metadata=metadata,
                                  global_step=runner.epoch, tag=self.tag)
