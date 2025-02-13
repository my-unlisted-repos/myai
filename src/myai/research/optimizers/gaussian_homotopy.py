# pylint:disable=signature-differs, not-callable #type:ignore
import torch
from torch.optim import Optimizer
import numpy as np

def flatten_params(params):
    return torch.cat([p.data.view(-1) for p in params])

def unflatten_params(flat_tensor, params):
    offset = 0
    for p in params:
        numel = p.data.numel()
        p.data.copy_(flat_tensor[offset:offset+numel].view(p.data.shape))
        offset += numel

class AdaptiveGaussianHomotopy(Optimizer):
    """adaptive tuning free gaussian homotopy should be globally convergent on non-stochastic non-convex functions
    but only if you get lucky.

    Args:
        params (_type_): params
        initial_sigma (float, optional): initial gaussian kernel sigma for convolving the objective. Defaults to 1.0.
        sigma_decay (float, optional): simga decay on convergence. Defaults to 0.5.
        num_perturbations (int, optional): number of pertubations more = better. Defaults to 20.
        convergence_threshold (_type_, optional):
            it is what it says. how it does that is in the code. Defaults to 1e-6.
        window_size (int, optional):
            window that tracks update norms to detect convergence and interacts with convergence
            theshold in unknown ways. Defaults to 10.
        max_grad_norm (float, optional): clips gradient norm for stability. Defaults to 10.0.
        lr (float, optional): initial leaerning rate its adaptive so no need to tune. Defaults to 1.0.
        adaptation_epsilon (_type_, optional): epsilon that you can tune without knowing what it does. Defaults to 1e-8.
        min_lr (_type_, optional): bounds lr from below. Defaults to 1e-6.
        max_lr (float, optional): bounds lr from above. Defaults to 10.0.
    """
    def __init__(self, params, initial_sigma=1.0, sigma_decay=0.5,
                 num_perturbations=20, convergence_threshold=1e-6,
                 window_size=10, max_grad_norm=10.0, lr=1.0,
                 adaptation_epsilon=1e-8, min_lr=1e-6, max_lr=10.0):
        defaults = dict(
            initial_sigma=initial_sigma,
            sigma_decay=sigma_decay,
            num_perturbations=num_perturbations,
            convergence_threshold=convergence_threshold,
            window_size=window_size,
            max_grad_norm=max_grad_norm,
            lr=lr,
            adaptation_epsilon=adaptation_epsilon,
            min_lr=min_lr,
            max_lr=max_lr
        )
        super().__init__(params, defaults)

        self._state = self.state[self.param_groups[0]['params'][0]]
        self._state.setdefault('sigma', initial_sigma)
        self._state.setdefault('perturbations', None)
        self._state.setdefault('convergence_buffer', [])
        self._state.setdefault('step_count', 0)
        self._state.setdefault('grad_history', [])
        self._state.setdefault('current_lr', lr)
        self._state.setdefault('best_loss', np.inf)

    @torch.no_grad
    def step(self, closure):
        group = self.param_groups[0]
        params = group['params']
        sigma = self._state['sigma']
        num_perturbations = group['num_perturbations']
        eps = group['adaptation_epsilon']
        original_flat = flatten_params(params)
        device = original_flat.device

        # Generate perturbations if needed
        if self._state['perturbations'] is None:
            n = original_flat.numel()
            self._state['perturbations'] = [
                torch.randn(n, device=device)
                for _ in range(num_perturbations)
            ]
            self._state['convergence_buffer'] = []
            print(f"Sigma: {sigma:.2e}, Generated {num_perturbations} perturbations")

        perturbations = self._state['perturbations']
        current_lr = self._state['current_lr']

        # Phase 1: Compute gradients and current convolved loss
        grad_matrix = []
        losses = []
        for epsilon in perturbations:
            # Apply perturbation
            perturbed_flat = original_flat + sigma * epsilon
            unflatten_params(perturbed_flat, params)

            # Compute loss and gradients
            self.zero_grad()
            with torch.enable_grad():
                loss = closure(False)
                loss.backward()
            losses.append(loss.item())

            # Capture gradient
            current_grad = torch.cat([p.grad.view(-1).detach().clone() for p in params])
            grad_matrix.append(current_grad)

            # Restore original parameters
            unflatten_params(original_flat, params)

        # Process gradient information
        grad_matrix = torch.stack(grad_matrix)
        avg_grad = grad_matrix.mean(dim=0)
        grad_var = grad_matrix.var(dim=0, unbiased=False)
        current_convolved_loss = np.mean(losses)

        # Compute preconditioned update direction
        preconditioner = 1.0 / (torch.sqrt(grad_var) + eps)
        update = avg_grad * preconditioner * sigma

        # Gradient clipping
        if group['max_grad_norm'] is not None:
            update_norm = update.norm()
            if update_norm > group['max_grad_norm']:
                update = update * (group['max_grad_norm'] / update_norm)

        # Phase 2: Trial step with adaptive learning rate
        trial_flat = original_flat - current_lr * update
        unflatten_params(trial_flat, params)

        # Compute new convolved loss
        new_losses = []
        for epsilon in perturbations:
            perturbed_trial = trial_flat + sigma * epsilon
            unflatten_params(perturbed_trial, params)
            with torch.no_grad():
                loss = closure(False).item()
            new_losses.append(loss)
            unflatten_params(trial_flat, params)
        new_convolved_loss = np.mean(new_losses)

        # Adaptive learning rate adjustment
        if new_convolved_loss < current_convolved_loss:
            # Accept step
            self._state['current_lr'] = min(current_lr * 2, group['max_lr'])
            self._state['best_loss'] = new_convolved_loss
        else:
            # Reject step and restore parameters
            unflatten_params(original_flat, params)
            self._state['current_lr'] = max(current_lr * 0.5, group['min_lr'])

        # Track parameter updates for convergence
        param_update_norm = (current_lr * update).norm().item()
        self._state['convergence_buffer'].append(param_update_norm)
        if len(self._state['convergence_buffer']) > group['window_size']:
            self._state['convergence_buffer'].pop(0)

        avg_update = np.mean(self._state['convergence_buffer'])

        # Sigma reduction logic
        if avg_update < group['convergence_threshold'] * sigma:
            self._state['sigma'] *= group['sigma_decay']
            self._state['perturbations'] = None
            self._state['current_lr'] = self._state['sigma']  # Reset learning rate
            print(f"Converged, reducing sigma to {self._state['sigma']:.2e}")

        self._state['step_count'] += 1
