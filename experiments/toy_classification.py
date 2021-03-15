import torch
import torch.nn as nn
from catalyst import dl
from models.net import Net
from models.transforms import ToTensor
from callabacks import DecisionBoundaryCallback
from runners import ClassificationRunner
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from experiment_utils import logger
from datasets import CirclesDataSet
from visualization_utils import visualize_dataset
import matplotlib.pyplot as plt

# dataset = CirclesDataSet(noise=0.3, minority_samples=50, transform=ToTensor(), target_transform=ToTensor())
# fig = visualize_dataset(dataset)
# fig.show()

x, y = make_classification(n_samples=10000,  # number of samples
                           n_features=2,  # feature/label count
                           n_informative=2,  # informative features
                           n_redundant=0,  # redundant features
                           n_repeated=0,  # duplicate features
                           class_sep=1.3,
                           n_clusters_per_class=1,  # number of clusters per class; clusters during plotting
                           weights=[0.99],  # proportions of samples assigned to each class
                           flip_y=0,  # fraction of samples whose class is assigned randomly.
                           random_state=21,
                           )
dataset = TensorDataset(torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.float32))

model = Net(n_samples=50)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.))
optimizer = {
    'classifier': torch.optim.Adam(model.classifier.parameters(), lr=0.02),
    'z': torch.optim.Adam([model.z], lr=0.01)
}

loaders = {
    "train": DataLoader(dataset, batch_size=64, shuffle=True),
    "valid": DataLoader(dataset, batch_size=64, shuffle=False)
}

runner = ClassificationRunner()
with logger('./logs', mode='debug') as log:
    runner.train(model=model,
                 criterion=criterion,
                 optimizer=optimizer,
                 loaders=loaders,
                 logdir=log,
                 num_epochs=10,
                 callbacks={
                     "criterion": dl.CriterionCallback(
                         metric_key="loss",
                         input_key="logits",
                         target_key="targets"),
                     "optimizer_z": dl.OptimizerCallback(metric_key="loss", optimizer_key='z'),
                     "visualization": DecisionBoundaryCallback()
                 },
                 )
