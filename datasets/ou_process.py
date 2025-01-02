import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset


def simulate_ou_process(T, len_trajectory, mu, sigma, theta, X_0):
    t = np.linspace(0,T,len_trajectory)
    X = np.zeros(len_trajectory)
    X[0] = X_0
    dt = T/len_trajectory

    for i in range(1,len_trajectory):
        dB = np.random.normal(0, np.sqrt(dt))
        drift = mu - X[i-1]
        diffusion = sigma
        X[i] = X[i-1] + theta * drift * dt + sigma * diffusion * dB
    return t, X


def create_ou_dataset(n_samples, T, len_trajectory, mu, sigma, theta, X_0):
    data = np.zeros((n_samples, 2, len_trajectory))
    for i in range(n_samples):
        t, X = simulate_ou_process(T, len_trajectory, mu, sigma, theta, X_0)
        data[i,0,:] = t
        data[i,1,:] = X
    
    data = torch.Tensor(data).permute(0,2,1)

    norm_T = torch.linspace(0,1,len_trajectory)
    spline_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data, norm_T)

    return data, spline_coeffs, norm_T

