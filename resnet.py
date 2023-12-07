import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import functional as F

class BatchResNet(nn.Module):
    def __init__(self, n_in, n_out, n, d, alpha=1/np.sqrt(2.), lmda = 1/np.sqrt(2.)):
        super(BatchResNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n = n
        self.d = d

        self.alpha = alpha
        self.lmda = lmda

        # Input layer
        self.input_scaling = torch.sqrt(torch.tensor(1. / self.n_in))

        # Hidden layers
        self.hidden_scaling = torch.sqrt(torch.tensor(2. / self.n))

        # Output layer
        self.output_scaling = torch.sqrt(torch.tensor(1. / self.n))

        self.weights = torch.randn((self.d + 1, self.n))
        self.out_weights = torch.randn(self.n_out)

    def forward(self, x):
        z = self.input_scaling * self.weights[0] * torch.norm(x)
        for i in range(self.d):
            z = self.lmda * self.weights[i+1] * torch.norm(torch.relu(z))* self.hidden_scaling + self.alpha * z
        
        return self.output_scaling * self.out_weights * torch.norm(z)

class ResNet(nn.Module):
    def __init__(self, n_in, n_out, n, d, alpha=1/np.sqrt(2.), lmda = 1/np.sqrt(2.)):
        super(ResNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n = n
        self.d = d

        self.alpha = alpha
        self.lmda = lmda

        self.input_scaling = torch.sqrt(torch.tensor(1. / self.n_in))
        self.hidden_scaling = torch.sqrt(torch.tensor(2. / self.n))
        self.output_scaling = torch.sqrt(torch.tensor(1. / self.n))
        
        # Input layer
        self.input_layer = nn.Linear(self.n_in, self.n, bias=False)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(self.d):
            self.hidden_layers.append(
                nn.Linear(self.n, self.n, bias=False)
            )

        # Output layer
        self.output_layer = nn.Linear(self.n, self.n_out, bias=False)

        self._init_weights()

    def forward(self, x):
        z = self.input_layer(x) * self.input_scaling
        for i in range(self.d):
            z = self.lmda*self.hidden_layers[i](F.relu(z))*self.hidden_scaling + self.alpha * z
        
        return self.output_layer(z) * self.output_scaling

    def _init_weights(self):
        nn.init.normal_(self.input_layer.weight, mean=0, std=1)

        for m in self.hidden_layers:
            nn.init.normal_(m.weight, mean=0, std=1)

        nn.init.normal_(self.output_layer.weight, mean=0, std=1)
        
