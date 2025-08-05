import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class SinCosTanODELinear(nn.Module):
    def __init__(self, hidden_size=128):
        super(SinCosTanODELinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size))
        self.c = nn.Parameter(torch.randn(hidden_size))
        self.d = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # Apply weird transformation
        x = torch.matmul(x, self.weights) + self.b * torch.sin(x) + self.c * torch.cos(x) + self.d * torch.tan(x)
        # Solve ODE using torchdiffeq
        def ode_func(t, y):
            return -y + t * x
        t = torch.linspace(0, 1, 100)
        y0 = torch.zeros_like(x)
        sol = odeint(ode_func, y0, t)
        return sol[-1]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            SinCosTanODELinear(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    net = Net()
    input_data = torch.randn(1, 784)
    output = net(input_data)
    print(output.shape)