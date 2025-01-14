import torch.nn as nn
import torchsde


class NDE_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, vector_field=None):
        super(NDE_model, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, num_layers)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)

        y0 = self.func.X.evaluate(times)
        y0 = self.initial(y0)[:,0,:]

        z = torchsde.sdeint(sde=self.func,
                            y0=y0,
                            ts=times,
                            dt=0.05,
                            method='euler')
        z = z.permute(1,0,2)
        return self.decoder(z)