import torch
import numpy as np
from torch.optim import Optimizer

class CrossEntropyMethod(Optimizer):
    def __init__(self, params, population_size=20, elite_frac=0.25,
                 sigma_init=0.1, sigma_decay=0.99, lr=1.0):
        """
        Args:
            params (iterable): iterable of parameters to optimize
            population_size (int): number of perturbations per step
            elite_frac (float): percentage of top samples to use for updates
            sigma_init (float): initial perturbation standard deviation
            sigma_decay (float): decay factor for perturbation std adaptation
            lr (float): learning rate for parameter updates
        """
        defaults = dict(population_size=population_size, elite_frac=elite_frac,
                        sigma_init=sigma_init, sigma_decay=sigma_decay, lr=lr)
        super().__init__(params, defaults)

        # Flatten parameters for easier handling
        self.param_shapes = []
        self.param_sizes = []
        self.param_numels = []
        self.param_device = None

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.param_shapes.append(p.shape)
                    self.param_sizes.append(p.size())
                    self.param_numels.append(p.numel())
                    if self.param_device is None:
                        self.param_device = p.device
                    else:
                        assert p.device == self.param_device, "All parameters must be on the same device"

        # Initialize distribution parameters
        self.means = [p.detach().clone().flatten() for group in self.param_groups for p in group['params'] if p.requires_grad]
        self.sigmas = [torch.full_like(m, sigma_init) for m in self.means]

        self.flat_means = torch.cat(self.means)
        self.flat_sigmas = torch.cat(self.sigmas)

        self.elite_size = max(1, int(population_size * elite_frac))
        self.total_params = sum(self.param_numels)
        self.population_size = population_size

    def _unflatten_params(self, flat_tensor):
        """Convert flat tensor to list of original-shaped parameters"""
        params = []
        offset = 0
        for numel, shape in zip(self.param_numels, self.param_shapes):
            params.append(flat_tensor[offset:offset+numel].view(shape))
            offset += numel
        return params

    def step(self, closure):
        """
        Performs a single optimization step

        Args:
            closure (callable): A closure that evaluates and returns the loss
        """
        # Generate population
        noise = torch.randn(self.population_size, self.total_params,
                           device=self.param_device)
        perturbations = self.flat_means + noise * self.flat_sigmas

        # Evaluate population
        losses = torch.zeros(self.population_size, device=self.param_device)
        with torch.no_grad():
            for i in range(self.population_size):
                sample_params = self._unflatten_params(perturbations[i])
                # Update model parameters
                for group, new_p in zip(self.param_groups, sample_params):
                    for p in group['params']:
                        if p.requires_grad:
                            p.copy_(new_p)
                losses[i] = closure(False)

        # Select elites
        elite_indices = torch.topk(-losses, self.elite_size).indices
        elite_samples = perturbations[elite_indices]

        # Update distribution parameters
        new_means = elite_samples.mean(dim=0)
        new_sigmas = elite_samples.std(dim=0, unbiased=False)

        # Apply learning rate and sigma decay
        self.flat_means = self.flat_means + self.param_groups[0]['lr'] * (new_means - self.flat_means)
        self.flat_sigmas = (self.param_groups[0]['sigma_decay'] * self.flat_sigmas +
                           (1 - self.param_groups[0]['sigma_decay']) * new_sigmas)

        # Update model parameters with new mean
        final_params = self._unflatten_params(self.flat_means)
        with torch.no_grad():
            for group, new_p in zip(self.param_groups, final_params):
                for p in group['params']:
                    if p.requires_grad:
                        p.copy_(new_p)

        return losses.mean()

    def _gather_parameters(self):
        """Collect current model parameters"""
        return torch.cat([p.detach().flatten() for group in self.param_groups for p in group['params'] if p.requires_grad])

    def _scatter_parameters(self, flat_params):
        """Distribute flat tensor to model parameters"""
        params = self._unflatten_params(flat_params)
        with torch.no_grad():
            for group, new_p in zip(self.param_groups, params):
                for p in group['params']:
                    if p.requires_grad:
                        p.copy_(new_p)