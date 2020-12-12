import torch
import numpy as np
from runners import CustomRunner, LogPRCurve
import catalyst.dl as dl
from catalyst.dl.callbacks import CheckpointCallback
from models.networks import TabularModel, MLP
from catalyst.contrib.nn import FocalLossBinary
from models.metrics import average_precision_metric
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import DataLoader
from models.transforms import ToTensor, ScalerTransform, Compose, OneHotTransform
from datasets import AdultDataSet, DatasetImbalanced

root = Path.cwd().parent / 'BenchmarkData' / 'adult'
adult_train = AdultDataSet(root, train=True,
                           target_transform=ToTensor())

scaler_transform = ScalerTransform(adult_train.data, features=adult_train.continuous_cols)
onehot_transform = OneHotTransform(adult_train.categorical_cols, adult_train.categories_sizes)
to_tensor = ToTensor()
feat_transform = Compose([scaler_transform, onehot_transform, to_tensor])
adult_train.transform = feat_transform
adult_test = AdultDataSet(root, train=False,
                          transform=feat_transform,
                          target_transform=ToTensor())

# Creating imbalanced adult dataset
adult_train = DatasetImbalanced(imbalance_ratio=None)(adult_train)
adult_test = DatasetImbalanced(imbalance_ratio=None)(adult_test)

loaders = {
    "train": DataLoader(adult_train, batch_size=32, shuffle=True),
    "valid": DataLoader(adult_test, batch_size=32, shuffle=True)
}

# model = TabularModel(adult_train.categorical_cols, adult_train.continuous_cols,
#                      adult_train.embeds, out_sz=1, layers=[100])
model = MLP(in_features=108, out_features=1, hidden_layers=[100, 50])

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Creating a loss function for training
criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(adult_train.pos_weight))
# criterion = FocalLossBinary(alpha=adult_train.pos_weight)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9, 20], gamma=0.3)

runner = CustomRunner()

callbacks = {
    "criterion": dl.CriterionCallback(
        input_key="targets",
        output_key="logits",
        prefix="loss"
    ),
    "Accuracy": dl.AccuracyCallback(input_key="targets",
                                    output_key="preds"),
    "mAP": dl.LoaderMetricCallback(input_key="targets",
                                   output_key="logits",
                                   prefix="mAP",
                                   metric_fn=average_precision_metric),
    "optimizer": dl.OptimizerCallback(
        metric_key="loss",
        accumulation_steps=1,
        grad_clip_params=None,
    ),
    "scheduler": dl.SchedulerCallback(mode="epoch"),
}
log_dir = "logs_tabular_basic"
runner.train(model=model,
             optimizer=optimizer,
             criterion=criterion,
             scheduler=scheduler,
             loaders=loaders,
             logdir=log_dir,
             num_epochs=5,
             callbacks=callbacks,
             main_metric="accuracy01",
             minimize_metric=False,
             verbose=True,
             load_best_on_end=True)

runner.infer(model=model,
             loaders={"infer": loaders["valid"]},
             callbacks=OrderedDict([
                 ("loader", CheckpointCallback(resume=f"{log_dir}/checkpoints/best.pth")),
                 ("pr_callback", LogPRCurve(log_dir=f"{log_dir}/infer_log"))
             ]))

# predictions = np.vstack(list(map(
#     lambda x: x.cpu().numpy(),
#     runner.predict_loader(loader=loaders["valid"], resume=f"{log_dir}/checkpoints/best.pth")
# )))
