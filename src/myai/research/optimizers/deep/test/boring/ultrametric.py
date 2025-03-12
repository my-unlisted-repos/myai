from collections import defaultdict

import torch
from torch.optim import Optimizer


class UltrametricOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99, 0.999), alphas=(0.6, 0.3, 0.1)):
        defaults = dict(lr=lr, betas=betas, alphas=alphas)
        super().__init__(params, defaults)

        # Initialize individual momentum buffers
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['individual_momentum'] = torch.zeros_like(p.data, requires_grad=False)

        # Layer and model momentum buffers
        self.layer_momentums = defaultdict(float)
        self.model_momentum = 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Update individual momentums
        individual_momentums = {}
        for group in self.param_groups:
            beta1 = group['betas'][0]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                im = state['individual_momentum']
                im.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                individual_momentums[p] = im.detach().clone()

        # Compute layer-wise averages
        layer_grads = defaultdict(list)
        for group in self.param_groups:
            for p in group['params']:
                if p in individual_momentums:
                    layer_grads[id(group)].append(individual_momentums[p].mean().item())

        for group in self.param_groups:
            gid = id(group)
            if gid in layer_grads:
                avg = sum(layer_grads[gid]) / len(layer_grads[gid])
                beta2 = group['betas'][1]
                self.layer_momentums[gid] = beta2 * self.layer_momentums[gid] + (1 - beta2) * avg

        # Compute model-wide average
        model_grads = [im.mean().item() for im in individual_momentums.values()]
        if model_grads:
            model_avg = sum(model_grads) / len(model_grads)
            beta3 = self.param_groups[0]['betas'][2]
            self.model_momentum = beta3 * self.model_momentum + (1 - beta3) * model_avg

        # Update parameters
        for group in self.param_groups:
            lr = group['lr']
            a1, a2, a3 = group['alphas']
            gid = id(group)
            layer_momentum = self.layer_momentums[gid]
            model_momentum = self.model_momentum

            for p in group['params']:
                if p.grad is None or p not in individual_momentums:
                    continue
                im = individual_momentums[p]
                update = a1 * im + a2 * layer_momentum + a3 * model_momentum
                p.data.add_(-lr * update)

        return loss