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

dataset_train = CirclesDataSet(noise=0.05, majority_samples=2000, minority_samples=100,
                               noisy_majority_samples=0, noisy_minority_samples=900,
                               transform=ToTensor(), target_transform=ToTensor())

dataset_valid = CirclesDataSet(train=False, noise=0.05, majority_samples=1000,
                               minority_samples=50, transform=ToTensor(), target_transform=ToTensor())

model = Net()

criterion = {
    "bce": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.)),
    }
optimizer = {
    'model': torch.optim.Adam(model.classifier.parameters(), lr=0.001),
}

loaders = {
    "train": DataLoader(dataset_train, batch_size=64, shuffle=True),
    "valid": DataLoader(dataset_valid, batch_size=64, shuffle=False),
}

with logger('./logs/circles/syn', mode='debug') as log:

    runner = ClassificationRunner(train_on_synthetic=True)
    checkpoint = utils.load_checkpoint(path="./logs/circles/bce/checkpoints/last.pth")
    utils.unpack_checkpoint(checkpoint=checkpoint, model=model)

    runner.train(model=model,
                 criterion=criterion,
                 optimizer=optimizer,
                 loaders=loaders,
                 logdir=log,
                 num_epochs=200,
                 hparams={"lr_z": 0.01,
                          "lr_meta": 0.001},
                 callbacks={
                     "visualization": DecisionBoundaryCallback(show=False),
                     "accuracy": dl.BatchMetricCallback(
                            metric=BalancedAccuracyMetric(), log_on_batch=False,
                            input_key="scores", target_key="targets",
                     ),
                     "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(log/'pr'),
                                                  loaders='valid'),
                     "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                                        input_key="scores",
                                                                                        target_key="targets"),
                                                  loaders='valid'
                                                  )

                 },
                 )
