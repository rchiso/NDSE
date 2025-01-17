import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.ou_process import create_ou_dataset
from sklearn.model_selection import train_test_split
import yaml


class Custom_Dataset(Dataset):
    """Torch template for custom datasets
    """
    def __init__(self, data, coeffs):
        self.data = data
        self.coeffs = coeffs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx, ...], self.coeffs[idx, ...],)
    

def OU(file='./datasets/OU_config.yaml'):
    """Generate data loaders for OU dataset

    Args:
        file (str, optional): Path to the OU config file. Defaults to './datasets/OU_config.yaml'.

    Returns:
        torch.utils.data.DataLoader: Training set data loader
        torch.utils.data.DataLoader: Test set data loader
    """
    with open(file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data, spline_coeffs = create_ou_dataset(config['n_samples'], config['T'], config['len_trajectory'], config['mu'], config['sigma'], config['theta'], config['X_0'])
    train_dataset, test_dataset, train_coeffs, test_coeffs = train_test_split(data, spline_coeffs, train_size=config['train_ratio'])
   
    train_data = Custom_Dataset(train_dataset, train_coeffs)
    test_data = Custom_Dataset(test_dataset, test_coeffs)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader






