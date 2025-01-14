import torch
import torch.nn as nn
import torchcde
from models.network import MLP

class NeuralMSDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NeuralMSDEFunc, self).__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal" # or "scalar"

        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim*2, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.noise_in = nn.Linear(1, hidden_dim)
        self.g_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def z_bar(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.linear_X(Xt)

        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        return self.emb(torch.cat([yy,Xt], dim=-1)), t
    
    def f(self, t, y):
        z, _ = self.z_bar(t,y)
        z = self.f_net(z)
        return self.linear_out(z)

    def g(self, t, y):
        z, t = self.z_bar(t,y)
        t = self.noise_in(t)
        return self.g_net(t) * z
    

class NeuralASDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NeuralASDEFunc, self).__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal" # or "scalar"

        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim*2, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.noise_in = nn.Linear(1, hidden_dim)
        self.g_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def z_bar(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.linear_X(Xt)

        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        return self.emb(torch.cat([yy,Xt], dim=-1)), t
    
    def f(self, t, y):
        z, _ = self.z_bar(t,y)
        z = self.f_net(z)
        return self.linear_out(z)

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        t = self.noise_in(t)
        return self.g_net(t)