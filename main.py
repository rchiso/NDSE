from train import train_model
import torch

if __name__ == '__main__':
    fig_path = './plots/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = './model_config.yaml'
    train_model(config_file, fig_path, device)
