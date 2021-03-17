import torch
import torch.nn as nn
from catalyst import dl
from models.net import Net
from catalyst import utils
from models.transforms import ToTensor
from callabacks import DecisionBoundaryCallback
from runners import ClassificationRunner
from torch.utils.data import TensorDataset, DataLoader
from experiment_utils import logger
from datasets import CirclesDataSet
from visualization_utils import visualize_dataset
import matplotlib.pyplot as plt

# dataset = CirclesDataSet(noise=0.3, minority_samples=50, transform=ToTensor(), target_transform=ToTensor())
# fig = visualize_dataset(dataset)
# fig.show()

torch.random.manual_seed(42)

dataset = CirclesDataSet(noise=0.1, minority_samples=50, transform=ToTensor(), target_transform=ToTensor())
minority = dataset.data[(dataset.target == 1).squeeze()]
model = Net(lr=0.001)
checkpoint = utils.load_checkpoint(path="./logs/circles/circles_imbalance/checkpoints/last.pth")
utils.unpack_checkpoint(checkpoint=checkpoint, model=model)
model.produce_samples(n_samples=50, minority=minority)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.))
optimizer = {
    'model': torch.optim.Adam(model.classifier.parameters(), lr=0.02),
    'z': torch.optim.Adam([model.z], lr=0.0001)
}

loaders = {
    "train": DataLoader(dataset, batch_size=64, shuffle=True),
    "valid": DataLoader(dataset, batch_size=64, shuffle=False)
}

runner = ClassificationRunner()
with logger('./logs/circles/syn', mode='debug') as log:
    runner.train(model=model,
                 criterion=criterion,
                 optimizer=optimizer,
                 loaders=loaders,
                 logdir=log,
                 num_epochs=100,
                 callbacks={
                     "criterion": dl.CriterionCallback(
                         metric_key="loss",
                         input_key="logits",
                         target_key="targets"),
                     # "optimizer_z": dl.OptimizerCallback(metric_key="loss", optimizer_key='z'),
                     "visualization": DecisionBoundaryCallback(plot_synthetic=True)
                 },
                 )
