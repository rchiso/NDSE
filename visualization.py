import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_predictions(epoch, test_loader, model, criterion, fig_path, device):
    """Plot the model predictions

    Args:
        epoch (int): Current epoch on the model training
        test_loader (torch.utils.data.DataLoader): Test set data loader
        model (NDE_model): Trained NDE model
        criterion (torch.nn.MSELoss): Loss function
        fig_path (str): Path to the directory to store plots
        device (str): Backend to be used
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in test_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)
    
            true = batch[0][:,:,1].to(device)
            pred = model(coeffs, times).squeeze(-1)
            loss = criterion(pred, true)
            total_loss += loss.item()
    
            all_preds.append(pred.cpu())
            all_trues.append(true.cpu())
    
    avg_loss = total_loss / len(test_loader)
    print(f'Epoch {epoch} Test Loss: {avg_loss}')
    
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    num_samples = 5      
    plt.figure(figsize=(8, 4))

    # Plot all samples without labels
    for i in range(num_samples):
        plt.plot(all_trues[i].numpy(), color='r')
        plt.plot(all_preds[i].numpy(), color='b')

    # Add a custom legend
    legend_elements = [
        Line2D([0], [0], color='r', label='True'),
        Line2D([0], [0], color='b', label='Predicted')
    ]
    plt.legend(handles=legend_elements)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim(-0.75, 1.25)
    plt.title(f'Model Predictions vs True Values (Epoch: {epoch})')
    plt.savefig(fig_path + f'Epoch={epoch}.png')

