from catalyst import utils
from datasets import TableDataset
from experiment_utils import load_config, get_model
from evaluation_utils import (
    evaluate_model_predictions,
    plot_train_valid_loss_graph,
    get_low_confidence_predictions,
)


def plot_loss_adult_ir_50():
    train_file = "experiments/adult_ir_50/base/logs/train.csv"
    valid_file = "experiments/adult_ir_50/base/logs/valid.csv"
    save_file = "experiments/adult_ir_50/base/loss.png"
    plot_train_valid_loss_graph(train_file, valid_file, save_file)


def save_evaluation_adult_ir_50():
    config = load_config('./experiments/adult_ir_50/config.yml')
    model = get_model(config.model, checkpoint='./experiments/adult_ir_50/logs/base/checkpoints/best.pth')
    data_train = TableDataset.from_npz(config.train_file)
    data_valid = TableDataset.from_npz(config.valid_file)
    evaluate_model_predictions(model, data_train, save='./experiments/adult_ir_50/logs/base/predict_train.csv')
    evaluate_model_predictions(model, data_valid, save='./experiments/adult_ir_50/logs/base/predict_valid.csv')


def save_low_confidence_adult_ir_50():
    get_low_confidence_predictions(data="./data/adult/ir50/adult_ir50.tra.npz",
                                   predictions="./experiments/adult_ir_50/base/predict_train.csv",
                                   save="./data/adult/ir50/low_minority_q50.npz")


if __name__ == "__main__":
    save_low_confidence_adult_ir_50()
