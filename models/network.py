import torch.nn as nn

class MLP(nn.Module):
    """Class implementing a simple fully connected MLP
    """
    def __init__(self, in_size, hidden_dim, out_size, num_layers):
        """Initializing the MLP

        Args:
            in_size (int): Size of the input layer
            hidden_dim (int): Size of the hidden layers
            out_size (int): Size of the output layer
            num_layers (int): Number of hidden layers
        """
        super().__init__()
        activation_fn = nn.ReLU()

        model = [nn.Linear(in_size, hidden_dim), activation_fn]
        for _ in range(num_layers - 1):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(activation_fn)
        model.append(nn.Linear(hidden_dim, out_size))
        self._model = nn.Sequential(*model)

    def forward(self, x):
        """Feed data into the network to get corresponding output

        Args:
            x (torch.Tensor): Input data, typically in a mini-batch 

        Returns:
            torch.Tensor: Output
        """
        return self._model(x)

