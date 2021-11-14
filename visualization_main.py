from visualization_utils import visualize_loss

if __name__ == "__main__":
    train_loss_file = "./experiments/adult_ir_50/logs/base/logs/train.csv"
    valid_loss_file = "./experiments/adult_ir_50/logs/base/logs/valid.csv"
    save_to_path = "./experiments/adult_ir_50/logs/base/logs/loss.png"
    fig = visualize_loss(train_loss_file, valid_loss_file)
    fig.savefig(save_to_path)
