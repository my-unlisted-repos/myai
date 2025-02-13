import torch
import torch.nn as nn

class FractalLayer(nn.Module):
    def __init__(self, num_iterations=5, hidden_size=128):
        super(FractalLayer, self).__init__()
        self.num_iterations = num_iterations
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        for _ in range(self.num_iterations):
            x = self.linear(x)
            x = x * x + x
            x = self.norm(x)
            x = self.activation(x)
        return x

class FractalNet(nn.Module):
    def __init__(self):
        super(FractalNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fractal = FractalLayer()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fractal(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    net = FractalNet()
    input_data = torch.randn(1, 784)
    output = net(input_data)
    print(output.shape)