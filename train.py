import torch
import torch.optim as optim
import yaml
from models.nde import NDE_model
from models.vector_fields import NeuralMSDEFunc, NeuralASDEFunc
from datasets.datasets import OU
from visualization import plot_predictions
import matplotlib.pyplot as plt


def train_model(config_file, fig_path, device):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if config['field'] == 'multiplicative':
        field = NeuralMSDEFunc
    elif config['field'] == 'additive':
        field = NeuralASDEFunc
    if config['data'] == 'OU':
        dataset = OU

    model = NDE_model(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'], num_layers=config['num_layers'], vector_field=field).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()
    train_loader, test_loader = dataset()

    plot_predictions(0, test_loader, model, criterion, fig_path, device)

    for epoch in range(1,config['num_epochs']+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            optimizer.zero_grad()
            true = batch[0][:,:,1].to(device)
            pred = model(coeffs, times).squeeze(-1)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Training Loss: {avg_loss}')

            plot_predictions(epoch, test_loader, model, criterion, fig_path, device)
            