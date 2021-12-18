from src.datasets import TableDataset
from src.evaluation_utils import get_low_confidence_predictions, evaluate_model_predictions
from src.experiment_utils import get_model


def test_get_low_confidence_predictions():
    data = './data/adult.tra.npz'
    predictions = "./experiment/results/predictions_train.csv"
    save_to = './experiment/results/train_conf_50.npz'
    results = get_low_confidence_predictions(data, predictions, save=save_to)


def test_evaluate_model_predictions(config):
    model = get_model(config.model, checkpoint="./experiment/models/model.pth")
    train_data = TableDataset.from_npz(config.train_file, name='train')
    evaluate_model_predictions(model, train_data, save_dir="./experiment/results")
    # evaluate_model(model, loader, save_to='./experiments/adult_ir_50/results/predictions_train.csv')
