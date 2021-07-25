import torch
import catalyst.dl as dl
from catalyst.dl import Experiment
from datasets import CirclesDataSet
from datasets import DatasetImbalanced
from models.transforms import ToTensor
import torch.nn as nn
from models.networks import CirclesHyperNetwork
from torch.utils.data import DataLoader
from runners import ClassificationRunner
from sklearn.metrics import balanced_accuracy_score
from models.metrics import average_precision_metric
from callabacks import HypernetVisualization


circles_train = CirclesDataSet(train=True, n_samples=1000, noise=0.05, transform=ToTensor(), target_transform=ToTensor())
circles_train = DatasetImbalanced(imbalance_ratio=0.1)(circles_train)
circles_test = CirclesDataSet(train=False, n_samples=1000, noise=0.05,  transform=ToTensor(), target_transform=ToTensor())
circles_test = DatasetImbalanced(imbalance_ratio=0.1)(circles_test)

classifier = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
generator = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
            )
encoder = nn.Identity()
hypernetwork = CirclesHyperNetwork(classifier)

criterion = {'bce': torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(circles_train.pos_weight)),
             'wbce': torch.nn.BCEWithLogitsLoss()}

model = {
         "hypernetwork": hypernetwork,
         "encoder": encoder}


optimizer = {
    "hyper_opt": torch.optim.Adam(model['hypernetwork'].parameters(), lr=0.0003, betas=(0.5, 0.999)),
    "syn_opt": torch.optim.Adam([model['hypernetwork'].x_syn], lr=0.03, betas=(0.9, 0.999))
}

logdir = './circles/hypernet2'

loaders = {
    "train": DataLoader(circles_train, batch_size=64, shuffle=True),
    "valid": DataLoader(circles_test, batch_size=64, shuffle=False),
}

experiment_train = Experiment(model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              loaders=loaders,
                              logdir=logdir,
                              main_metric="mAP",
                              minimize_metric=False,
                              num_epochs=2,
                              callbacks=[
                                  dl.CriterionCallback(criterion_key='wbce',
                                                       prefix='loss_syn'),
                                  dl.LoaderMetricCallback(input_key="targets",
                                                          output_key="logits",
                                                          prefix="mAP",
                                                          metric_fn=average_precision_metric),
                                  dl.LoaderMetricCallback(input_key="targets",
                                                          output_key="preds",
                                                          prefix="BalancedAccuracy",
                                                          metric_fn=balanced_accuracy_score),
                                  dl.OptimizerCallback(loss_key='loss_syn',
                                                       optimizer_key='syn_opt'),
                                  dl.OptimizerCallback(loss_key='loss_h',
                                                       optimizer_key='hyper_opt'),
                                  HypernetVisualization(),
                                  dl.TensorboardLogger()
                              ])


runner = ClassificationRunner(use_hyper_network=True)
runner.run_experiment(experiment_train)