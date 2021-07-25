import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.metrics import BalancedAccuracyMetric, APMetric
from callabacks import LogPRCurve
from catalyst import dl
from catalyst import utils
from runners import ClassificationRunner, MetaClassificationRunner
from experiment_utils import experiment_logger
from models.net import Net
from datasets import TableDataset
from catalyst.data.sampler import BalanceClassSampler
from data_utils import SyntheticDataset

torch.manual_seed(42)
hparams = {"ir": 10,
           "lr_model": 1e-3,
           "batch_size": 64,
           }

train_file = './data/ir200/breast_train_ir200.npz'
# complement_train_file = 'data/ir10/complement.npz'
test_file = './data/full/breast_test.npz'

train_data = TableDataset.from_npz(train_file, train=True)
test_data = TableDataset.from_npz(test_file, train=False)

classifier = nn.Sequential(nn.Linear(30, 64), nn.ReLU(),
                           nn.Linear(64, 64), nn.ReLU(),
                           nn.Linear(64, 1))
model = Net(classifier)

criterion = {
    "bce": nn.BCEWithLogitsLoss(),
    "bce_weighted": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hparams["ir"]))
}

optimizer = {
    'model': torch.optim.Adam(model.classifier.parameters(), lr=hparams["lr_model"]),
}

samplers = {
    "upsampling": BalanceClassSampler(train_data.target, mode='upsampling'),
    "downsampling": BalanceClassSampler(train_data.target, mode='downsampling')

}

scheduler = {
    "lr_scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer["model"], [19])
}

loaders = {
    "train": DataLoader(train_data, batch_size=hparams['batch_size'], shuffle=True),
    "valid": DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False)
}

with experiment_logger('./logs/ir200/base', mode='debug') as logger:
    # Logging the data for experiment
    logger.log_data(f"Train data shape: {train_data.data.shape[0], train_data.data.shape[1]}")
    logger.log_data(f"Train minority size: {int(train_data.target.sum())}")
    logger.log_data(f"Test data shape: {test_data.data.shape[0], test_data.data.shape[1]}")
    logger.log_data(f"Test data minority size: {int(test_data.target.sum())}")

    runner = ClassificationRunner()
    runner.train(model=model,
                 criterion=criterion["bce"],
                 optimizer=optimizer["model"],
                 scheduler=scheduler["lr_scheduler"],
                 loaders=loaders,
                 logdir=logger.logdir,
                 num_epochs=200,
                 hparams=hparams,
                 valid_loader="valid",
                 valid_metric="ap",
                 minimize_valid_metric=False,
                 callbacks={
                     "criterion": dl.CriterionCallback(metric_key="loss_x", input_key="logits", target_key="targets"),
                     "optimizer": dl.OptimizerCallback(metric_key="loss_x"),
                     # "scheduler": dl.SchedulerCallback(),
                     "accuracy": dl.BatchMetricCallback(
                         metric=BalancedAccuracyMetric(), log_on_batch=False,
                         input_key="scores", target_key="targets",
                     ),
                     "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(os.path.join(logger.logdir, 'pr')),
                                                  loaders='valid'),
                     "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                                        input_key="scores",
                                                                                        target_key="targets"),
                                                  loaders='valid'
                                                  )

                 },
                 )





