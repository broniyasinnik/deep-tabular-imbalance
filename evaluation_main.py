import pandas as pd
from pathlib import Path
from evaluation_utils import plot_losses

EXPERIMENT_LOG_DIR = Path("./experiments/adult_ir_50/meta1/logs")
SAVE_DIR = Path("./experiments/adult_ir_50/meta1/")

if __name__ == "__main__":
    train_log = pd.read_csv(EXPERIMENT_LOG_DIR / "train.csv")
    valid_log = pd.read_csv(EXPERIMENT_LOG_DIR / "valid.csv")
    plot_losses(dict(
        train_loss=train_log["loss"].to_numpy(),
        emulative_loss=train_log["loss_emulative"].to_numpy(),
    ),
        save=SAVE_DIR / "loss.png")
