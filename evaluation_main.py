from torch.utils.data import DataLoader
from experiment_utils import get_config, get_model, get_train_data


if __name__ == "__main__":
    config = get_config('./experiments/adult_ir_50/config.yml')
    model = get_model(config.model, checkpoint='./experiments/adult_ir_50/logs/base/checkpoints/best.pth')
    train_data = get_train_data(config.train_file)
    loader = DataLoader(train_data, shuffle=False)
    evaluate_model(model, loader, save_to='./experiments/adult_ir_50/results/predictions_train.csv')

