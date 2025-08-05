import torch
import torch.nn as nn
import torch.nn.functional as F

class LorenzLayer(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(hidden_size))
        self.rho = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Apply Lorenz attractor equations to each feature channel
        dxdt = self.sigma[:x.shape[1]-1] * (x[:, 1:] - x[:, :-1])
        dydt = x[:, :-1] * (self.rho[:x.shape[1]-1] - x[:, 1:]) - x[:, 1:]
        dzdt = x[:, :-1] * x[:, 1:] - self.beta[:x.shape[1]-1] * x[:, 1:]
        # Update x using Euler's method
        x[:, :-1] = x[:, :-1] + dxdt * 0.01
        x[:, 1:] = x[:, 1:] + (dydt + dzdt) * 0.01
        return x

class LorenzLayer2(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(hidden_size))
        self.rho = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Apply Lorenz attractor equations to each feature channel
        dxdt = self.sigma * x
        dydt = self.rho * x - x**2
        dzdt = x * self.beta - x**2
        # Update x using Euler's method
        x = x + (dxdt + dydt + dzdt) * 0.01
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            LorenzLayer2(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    net = Net()
    input_data = torch.randn(1, 784)
    output = net(input_data)
    print(output.shape)