"""Configurable MLP with domain-based input scaling and linear regression head."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(10,), activation='tanh'):
        super().__init__()
        activations = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid}
        if activation not in activations or not hidden_sizes or any(
                not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in hidden_sizes):
            raise ValueError('Provide positive hidden widths and tanh/relu/sigmoid activation')
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation
        self.register_buffer('input_scale', torch.tensor([2., 4.]))
        layers = []
        previous = 2
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), activations[activation]()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x / self.input_scale)
