import torch
from torch import nn
import torch.nn.functional as F

class FourierBasisLayer(nn.Module):
    """works like linear"""
    def __init__(self, in_channels, out_channels, num_bases=4):
        super().__init__()
        self.input_dim = in_channels
        self.output_dim = out_channels
        self.num_bases = num_bases  # K basis functions per neuron

        # Parameters for each basis function in each output neuron
        # Weights: (output_dim, num_bases, input_dim)
        self.w = nn.Parameter(torch.randn(out_channels, num_bases, in_channels))
        # Offsets: (output_dim, num_bases)
        self.c = nn.Parameter(torch.randn(out_channels, num_bases))
        # Phases: (output_dim, num_bases)
        self.d = nn.Parameter(torch.randn(out_channels, num_bases))
        # Amplitudes: (output_dim, num_bases)
        self.a = nn.Parameter(torch.randn(out_channels, num_bases))

        # Initialize parameters
        nn.init.normal_(self.w, mean=0, std=0.1)  # Small initial frequencies
        nn.init.constant_(self.c, 0)
        nn.init.uniform_(self.d, 0, 2 * torch.pi)  # Phase between 0-2π
        nn.init.normal_(self.a, mean=0, std=0.1)    # Small initial amplitudes

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)

        # Expand dimensions for broadcasting
        x = x.view(batch_size, 1, 1, self.input_dim)  # (B, 1, 1, D)
        w = self.w.view(1, self.output_dim, self.num_bases, self.input_dim)  # (1, O, K, D)
        c = self.c.view(1, self.output_dim, self.num_bases, 1)  # (1, O, K, 1)
        d = self.d.view(1, self.output_dim, self.num_bases, 1)  # (1, O, K, 1)

        # Compute z = w^T x + c for all bases and neurons
        z = torch.sum(w * x, dim=-1) + c.squeeze(-1)  # (B, O, K)

        # Apply sine with phase shift
        s = torch.sin(z + d.squeeze(-1))  # (B, O, K)

        # Weight by amplitudes
        a = self.a.view(1, self.output_dim, self.num_bases)  # (1, O, K)
        outputs = torch.sum(a * s, dim=-1)  # (B, O)

        return outputs