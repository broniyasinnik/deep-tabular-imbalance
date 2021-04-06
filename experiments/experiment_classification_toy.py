import torch
import torch.nn as nn
from catalyst import dl
from models.net import Net
from models.metrics import BalancedAccuracyMetric, APMetric
from catalyst import utils
from models.transforms import ToTensor
from callabacks import DecisionBoundaryCallback, LogPRCurve
from runners import ClassificationRunner
from torch.utils.data import TensorDataset, DataLoader
from experiment_utils import logger
from datasets import CirclesDataSet
import warnings

warnings.filterwarnings("ignore")

torch.random.manual_seed(42)
hparams = {"train":
               {"majority_samples": 5000,
                "minority_samples": 50,
                "noisy_minority_samples": 200,
                "noisy_majority_samples": 0,
                },
           "valid": {"majority_samples": 2500,
                     "minority_samples": 25,
                     },
           "ir": 100,
           "noise": 0.09,
           "model_lr": 0.0001,
           "lr_z": 1,
           "lr_meta": 0.001
           }


dataset_train = CirclesDataSet(noise=hparams["noise"],
                               majority_samples=hparams["train"]["majority_samples"],
                               minority_samples=hparams["train"]["minority_samples"],
                               noisy_majority_samples=hparams["train"]["noisy_majority_samples"],
                               noisy_minority_samples=hparams["train"]["noisy_minority_samples"],
                               transform=ToTensor(), target_transform=ToTensor())

dataset_valid = CirclesDataSet(train=False, noise=hparams["noise"],
                               majority_samples=hparams["valid"]["majority_samples"],
                               minority_samples=hparams["valid"]["minority_samples"],
                               transform=ToTensor(),
                               target_transform=ToTensor())

model = Net()

criterion = {
    "bce": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hparams["ir"])),
    }
optimizer = {
    'model': torch.optim.Adam(model.classifier.parameters(), lr=hparams["model_lr"]),
}

loaders = {
    "train": DataLoader(dataset_train, batch_size=64, shuffle=True),
    "valid": DataLoader(dataset_valid, batch_size=64, shuffle=False),
}

with logger('./logs/circles/ir100/syn', mode='debug') as log:
    runner = ClassificationRunner(train_on_synthetic=True)
    checkpoint = utils.load_checkpoint(path="logs/circles/ir100/bce/checkpoints/last.pth")
    utils.unpack_checkpoint(checkpoint=checkpoint, model=model)

    runner.train(model=model,
                 criterion=criterion,
                 optimizer=optimizer,
                 loaders=loaders,
                 logdir=log,
                 num_epochs=200,
                 hparams=hparams,
                 callbacks={
                     "visualization": DecisionBoundaryCallback(show=False),
                     "accuracy": dl.BatchMetricCallback(
                         metric=BalancedAccuracyMetric(), log_on_batch=False,
                         input_key="scores", target_key="targets",
                     ),
                     "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(log / 'pr'),
                                                  loaders='valid'),
                     "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                                        input_key="scores",
                                                                                        target_key="targets"),
                                                  loaders='valid'
                                                  )

                 },
                 )
