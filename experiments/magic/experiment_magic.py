import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.metrics import BalancedAccuracyMetric, APMetric
from callabacks import LogPRCurve
from catalyst import dl
from catalyst import utils
from runners import ClassificationRunner
from data_utils import DatasetImbalanced
from experiment_utils import logger
from models.net import Net
from datasets import TableDataset
from catalyst.data.sampler import BalanceClassSampler

hparams = {"ir": 1,
           "model_lr": 1e-4,
           }

train_file = './data/full/magic_train.npz'
test_file = './data/ir100/magic_test_ir_100.npz'

train_data = TableDataset.from_npz(train_file, train=True)
test_data = TableDataset.from_npz(test_file, train=False)


classifier = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                           nn.Linear(32, 32), nn.ReLU(),
                           nn.Linear(32, 1))


model = Net(classifier)

criterion = {
    "bce": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hparams["ir"])),
}

optimizer = {
    'model': torch.optim.Adam(model.classifier.parameters(), lr=hparams["model_lr"]),
}

samplers = {
    "upsampling": BalanceClassSampler(train_data.target, mode='upsampling'),
    "downsampling": BalanceClassSampler(train_data.target, mode='downsampling')

}

loaders = {
    "train": DataLoader(train_data, batch_size=128, shuffle=True),
    "valid": DataLoader(test_data, batch_size=128, shuffle=False)
}

with logger('./logs/ir100/potential', mode='debug') as log:
    runner = ClassificationRunner(train_on_synthetic=False)
    # checkpoint = utils.load_checkpoint(path="logs/ir30/baseline/checkpoints/best.pth")
    # utils.unpack_checkpoint(checkpoint=checkpoint, model=model)

    runner.train(model=model,
                 criterion=criterion,
                 optimizer=optimizer,
                 loaders=loaders,
                 logdir=log,
                 num_epochs=500,
                 hparams=hparams,
                 valid_loader="valid",
                 valid_metric="ap",
                 minimize_valid_metric=False,
                 callbacks={
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
