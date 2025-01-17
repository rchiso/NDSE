from train import train_model
import torch

if __name__ == '__main__':
    fig_path = './plots/' #Path to the directory to store plots
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Backend to be used
    config_file = './model_config.yaml' #Model configuration parameters
    train_model(config_file, fig_path, device)
