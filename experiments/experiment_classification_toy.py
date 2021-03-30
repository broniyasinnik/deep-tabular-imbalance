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

torch.random.manual_seed(42)

dataset_train = CirclesDataSet(noise=0.05, minority_samples=100, transform=ToTensor(), target_transform=ToTensor())
dataset_valid = CirclesDataSet(train=False, noise=0.05, majority_samples=500,
                               minority_samples=50, transform=ToTensor(), target_transform=ToTensor())

minority = dataset_train.data[(dataset_train.target==1).squeeze()]
model = Net(lr=0.2)
checkpoint = utils.load_checkpoint(path="./logs/circles/imbalance/checkpoints/last.pth")
utils.unpack_checkpoint(checkpoint=checkpoint, model=model)
model.produce_samples(n_samples=100, minority=minority)

criterion = {
    # "bce": nn.BCEWithLogitsLoss(),
    "bce": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.)),
    "kl": nn.KLDivLoss()
    }
optimizer = {
    'model': torch.optim.Adam(model.classifier.parameters(), lr=0.0001),
    'z': torch.optim.SGD([model.z], lr=0.001)
}

loaders = {
    "train": DataLoader(dataset_train, batch_size=64, shuffle=True),
    "valid": DataLoader(dataset_valid, batch_size=64, shuffle=False),
}
runner = ClassificationRunner()
with logger('./logs/circles/meta', mode='debug') as log:
    runner.train(model=model,
                 criterion=criterion,
                 optimizer=optimizer,
                 loaders=loaders,
                 logdir=log,
                 num_epochs=100,
                 callbacks={
                     "criterion": dl.CriterionCallback(
                         criterion_key="bce",
                         metric_key="loss",
                         input_key="logits",
                         target_key="targets"),
                     # "optimizer": dl.OptimizerCallback(metric_key="loss", optimizer_key='model'),
                     "visualization": DecisionBoundaryCallback(plot_synthetic=True),
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
