import pandas as pd
import torch
import torch.nn as nn
from catalyst import dl
from catalyst import utils
from experiment_utils import logger
from runners import ClassificationRunner
from torch.utils.data import DataLoader
from callabacks import LogPRCurve
from models.net import Net
from models.metrics import BalancedAccuracyMetric, APMetric
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datasets import TableDataset
from data_utils import DatasetImbalanced, SyntheticDataset


torch.random.manual_seed(42)
df = pd.read_csv("../../BenchmarkData/spine/Dataset_spine.csv")
df = df.drop('Unnamed: 13', axis=1)
df['Class_att'] = df['Class_att'].astype('category')
encode_map = {
    'Abnormal': 1,
    'Normal': 0
}


df['Class_att'].replace(encode_map, inplace=True)
X = df.iloc[:, 0:-1].to_numpy()
y = df.iloc[:, -1].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


hparams = {"ir": 5.,
           "model_lr": 1e-3,
           "lr_z": 0.2,
           "lr_meta": 0.001
           }

train_full = TableDataset.from_tensors(X_train, y_train, train=True)
test_full = TableDataset.from_tensors(X_test, y_test, train=False)
train_imb = DatasetImbalanced(train_full, imbalance_ratio=1./hparams["ir"])
test_imb = DatasetImbalanced(test_full, imbalance_ratio=1./hparams["ir"])
train_imb_syn = SyntheticDataset(train_imb, n_synthetic_minority=25, n_synthetic_majority=0)

classifier = nn.Sequential(nn.Linear(12, 64), nn.ReLU(),
                            nn.Linear(64, 64), nn.ReLU(),
                            nn.Linear(64, 1))
model = Net(classifier)

criterion = {
    "bce": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hparams["ir"])),
    }
optimizer = {
    'model': torch.optim.Adam(model.classifier.parameters(), lr=hparams["model_lr"]),
}

loaders = {
    "train": DataLoader(train_imb_syn, batch_size=64, shuffle=True),
    "valid": DataLoader(test_imb, batch_size=64, shuffle=False),
}

with logger('./logs/spine/debug', mode='debug') as log:
    runner = ClassificationRunner(train_on_synthetic=True)
    checkpoint = utils.load_checkpoint(path="logs/spine/ir5/checkpoints/best.pth")
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