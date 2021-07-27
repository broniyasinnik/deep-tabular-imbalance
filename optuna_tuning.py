import optuna
import torch
from torch import nn
from catalyst import dl
import numpy as np
from sklearn.model_selection import train_test_split
from models.metrics import APMetric
from runners import ClassificationRunner
from experiment_utils import prepare_mlp_classifier
from torch.utils.data import DataLoader
from datasets import TableDataset

def objective_meta(trial):
    ...

def objective_base(trial):
    lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
    num_hidden = int(trial.suggest_loguniform("num_hidden", 32, 128))
    loaders = {
        "train": DataLoader(train_data, batch_size=64, shuffle=True),
        "valid": DataLoader(valid_data, batch_size=64, shuffle=False)
    }
    classifier = prepare_mlp_classifier(input_dim=X.shape[1], hidden_dims=num_hidden)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    runner = ClassificationRunner()
    runner.train(
        model=classifier,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks={
            "ap": dl.LoaderMetricCallback(metric=APMetric(),
                                          input_key="scores",
                                          target_key="targets"),
            "optuna": dl.OptunaPruningCallback(
                loader_key="valid", metric_key="ap", minimize=False, trial=trial
            ),
        },
        num_epochs=200,
    )
    score = trial.best_score
    return score


def run_study():
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1, n_warmup_steps=0, interval_steps=1
        ),
    )
    study.optimize(objective_base, n_trials=20, timeout=300)
    print(study.best_value, study.best_params)


if __name__ == "__main__":
    train_file = './Keel1/winequality-red-4/winequality-red-4.tra.npz'
    validation_size = 0.2
    data = np.load(train_file)
    seed = 42
    X, y = data["X"], data["y"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y,
                                                          test_size=validation_size, random_state=seed)
    train_data = TableDataset(features=X_train, targets=y_train)
    valid_data = TableDataset(features=X_valid, targets=y_valid)
    run_study()
