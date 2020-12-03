import torch
from runners import CustomRunner
from catalyst import dl
from catalyst.callbacks.metrics.precision import AveragePrecisionCallback
from models.metrics import f1_score
from models.networks import TabularModel
from pathlib import Path
from torch.utils.data import DataLoader
from models.transforms import ToTensor, ScalerTransform, Compose
from datasets import AdultDataSet

root = Path.cwd().parent / 'BenchmarkData' / 'adult'
adult_train = AdultDataSet(root, train=True,
                           target_transform=ToTensor())

feat_transform = Compose([ScalerTransform(adult_train.data, features=adult_train.continuous_cols), ToTensor()])
adult_train.transform = feat_transform

adult_test = AdultDataSet(root, train=False,
                          transform=feat_transform,
                          target_transform=ToTensor())

loaders = {
    "train": DataLoader(adult_train, batch_size=32, drop_last=True),
    "valid": DataLoader(adult_test, batch_size=32)
}

model = TabularModel(adult_train.embeds, adult_test.num_continuous, out_sz=1, layers=[100])

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCELoss()
runner = CustomRunner()

callbacks = {
    "criterion": dl.CriterionCallback(
        input_key="targets",
        output_key="logits",
        prefix="loss"
    ),
    "metric_f1": dl.LoaderMetricCallback(prefix="f1",
                            metric_fn=f1_score,
                            input_key="targets",
                            output_key="logits"),
    "metric_AP": AveragePrecisionCallback(prefix="auc",
                              input_key="targets",
                              output_key="logits",
                              class_args=['50k>']),
    "optimizer": dl.OptimizerCallback(
         metric_key="loss",
         accumulation_steps=1,
         grad_clip_params=None,
    )

}
runner.train(model=model,
             optimizer=optimizer,
             criterion=criterion,
             loaders=loaders,
             logdir='./logs_tabular_basic',
             num_epochs=5,
             callbacks=callbacks,
             verbose=True,
             load_best_on_end=True)
