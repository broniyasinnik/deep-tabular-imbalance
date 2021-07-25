import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.metrics import BalancedAccuracyMetric, APMetric
from callabacks import LogPRCurve
from catalyst import dl
from catalyst import utils
from runners import ClassificationRunner, MetaClassificationRunner
from experiment_utils import logger
from models.net import Net
from datasets import TableDataset
from catalyst.data.sampler import BalanceClassSampler
from data_utils import SyntheticDataset

torch.manual_seed(42)
hparams = {"ir": 30,
           "lr_model": 1e-3,
           "lr_z": 0,
           "batch_size": 128,
           "synthetic_batch_size": 32,
           "holdout_batch_size": 64,
           "lr_meta": 1e-5,
           }

train_file = './data/ir30/cardio_train.npz'
holdout_file = './data/ir30/cardio_holdout.npz'
complement_train_file = 'data/ir30/cardio_train_complement.npz'
test_file = './data/ir30/cardio_test_ir_30.npz'

train_data = TableDataset.from_npz(train_file, train=True)
holdout_data = TableDataset.from_npz(holdout_file, train=False)
complement_train = TableDataset.from_npz(complement_train_file, train=True)
test_data = TableDataset.from_npz(test_file, train=False)

print("Train data shape:", train_data.data.shape)
print("Train minority size:", train_data.target.sum())
print("Hold out shape:", holdout_data.data.shape)
print("Hold out minority size:", holdout_data.target.sum())
print("Test data shape:", test_data.data.shape)
print("Test data minority size:", test_data.target.sum())


# train_imb = DatasetImbalanced(train_full, imbalance_ratio=1./hparams["ir"])
# test_imb = DatasetImbalanced(test_full, imbalance_ratio=1./hparams["ir"])
n_synthetic = int(train_data.target.sum())
synthetic_data = SyntheticDataset(train_data, complement_dataset=complement_train,
                              n_synthetic_minority=128, n_synthetic_majority=128)


classifier = nn.Sequential(nn.Linear(15, 64), nn.ReLU(),
                           nn.Linear(64, 64), nn.ReLU(),
                           nn.Linear(64, 1))
model = Net(classifier)

criterion = {
    "bce": nn.BCEWithLogitsLoss()
    # "bce": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hparams["ir"])),
}
optimizer = {
    'model': torch.optim.Adam(model.classifier.parameters(), lr=hparams["lr_model"]),
}

samplers = {
    "upsampling_train": BalanceClassSampler(train_data.target, mode='upsampling'),
    "upsampling_holdout": BalanceClassSampler(holdout_data.target, mode='upsampling'),
    "downsampling": BalanceClassSampler(train_data.target, mode='downsampling')

}

loaders = {
    # "train": DataLoader(train_data, batch_size=128, shuffle=True),
    # "train": DataLoader(train_data, batch_size=hparams['batch_size'], sampler=samplers["upsampling"]),
    "train": DataLoader(synthetic_data, batch_size=hparams["synthetic_batch_size"], shuffle=True),
    "holdout": DataLoader(holdout_data, batch_size=hparams['holdout_batch_size'], sampler=samplers["upsampling_holdout"]),
    "valid": DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False)
}

runners = {
    "classification": ClassificationRunner(),
    "metaclassification": MetaClassificationRunner()
}

with logger('./logs/ir30/debug', mode='debug') as log:
    runner = runners["metaclassification"]
    checkpoint = utils.load_checkpoint(path="logs/ir30/baseline/upsampling/checkpoints/best.pth")
    utils.unpack_checkpoint(checkpoint=checkpoint, model=model)

    runner.train(model=model,
                 criterion=criterion,
                 optimizer=optimizer,
                 loaders=loaders,
                 logdir=log,
                 num_epochs=200,
                 hparams=hparams,
                 valid_loader="valid",
                 valid_metric="ap",
                 minimize_valid_metric=False,
                 callbacks={
                     "loaders": dl.PeriodicLoaderCallback(valid_loader_key='valid',
                                                          valid_metric_key='ap',
                                                          minimize=False,
                                                          synthetic=0,
                                                          holdout=0),
                     # "accuracy": dl.BatchMetricCallback(
                     #     metric=BalancedAccuracyMetric(), log_on_batch=False,
                     #     input_key="scores", target_key="targets",
                     # ),
                     "pr": dl.ControlFlowCallback(base_callback=LogPRCurve(log / 'pr'),
                                                  loaders='valid'),
                     "ap": dl.ControlFlowCallback(base_callback=dl.LoaderMetricCallback(metric=APMetric(),
                                                                                        input_key="scores",
                                                                                        target_key="targets"),
                                                  loaders='valid'
                                                  )

                 },
                 )





