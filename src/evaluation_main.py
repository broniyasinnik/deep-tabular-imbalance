from pathlib import Path

import pandas as pd

from evaluation_utils import plot_losses, save_pr_figure

DATASET_NAME = "adult_ir_50"
EXPERIMENT_NAME = "meta"
EXPERIMENT_LOG_DIR = Path("../experiments/") / DATASET_NAME / EXPERIMENT_NAME / "logs"
EXPERIMENT_RESULT_DIR = Path("../experiments/") / DATASET_NAME / EXPERIMENT_NAME / "results"


def evaluate_emulative():
    train_log = pd.read_csv(EXPERIMENT_LOG_DIR / "train.csv")
    train_predictions = pd.read_csv(EXPERIMENT_RESULT_DIR / "train_predictions.csv")
    valid_log = pd.read_csv(EXPERIMENT_LOG_DIR / "valid.csv")
    valid_predictions = pd.read_csv(EXPERIMENT_RESULT_DIR / "valid_predictions.csv")
    plot_losses(
        dict(
            train_loss=train_log["loss"].to_numpy(),
            emulative_loss=train_log["loss_emulative"].to_numpy(),
        ),
        save=EXPERIMENT_RESULT_DIR / "loss.png",
    )
    prediction_dict = {
        EXPERIMENT_NAME: {
            "labels": valid_predictions["labels"].to_numpy(),
            "scores": valid_predictions["scores"].to_numpy(),
        }
    }
    save_pr_figure(prediction_dict, EXPERIMENT_RESULT_DIR)


def evaluate_baseline():
    train_log = pd.read_csv(EXPERIMENT_LOG_DIR / "train.csv")
    train_predictions = pd.read_csv(EXPERIMENT_RESULT_DIR / "train_predictions.csv")
    valid_log = pd.read_csv(EXPERIMENT_LOG_DIR / "valid.csv")
    valid_predictions = pd.read_csv(EXPERIMENT_RESULT_DIR / "valid_predictions.csv")
    plot_losses(
        dict(
            train_loss=train_log["loss"].to_numpy(),
            valid_loss=valid_log["loss"].to_numpy(),
        ),
        save=EXPERIMENT_RESULT_DIR / "loss.png",
    )
    prediction_dict = {
        EXPERIMENT_NAME: {
            "labels": valid_predictions["labels"].to_numpy(),
            "scores": valid_predictions["scores"].to_numpy(),
        }
    }
    save_pr_figure(prediction_dict, EXPERIMENT_RESULT_DIR)


if __name__ == "__main__":
    evaluate_emulative()
