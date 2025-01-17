import torch.nn as nn
import torchsde


class NDE_model(nn.Module):
    """ Class implementing a simple neural network architecture for differential equations

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, vector_field=None):
        """Initialzing the NDE architecture

        Args:
            input_dim (int): Size of the input layer
            hidden_dim (int): Size of the hidden layer(s)
            output_dim (_type_): Size of the output layer
            num_layers (_type_): Number of hidden layers
            vector_field (_type_, optional): The vector field used in solving the SDE. Defaults to None.
        """
        super(NDE_model, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, num_layers)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, coeffs, times):
        """Feed forward input data to get output

        Args:
            coeffs (torch.Tensor): Coefficients of the hermite cubic interpolation polynomial
            times (torch.Tensor): Time points of the path, compressed in the range [0,1]

        Returns:
            torch.Tensor: Output
        """
        self.func.set_X(coeffs, times)

        y0 = self.func.X.evaluate(times)
        y0 = self.initial(y0)[:,0,:] #Model for starting condition

        z = torchsde.sdeint(sde=self.func,
                            y0=y0,
                            ts=times,
                            dt=0.05,
                            method='euler')
        z = z.permute(1,0,2)
        return self.decoder(z)