# pylint:disable=signature-differs, not-callable

import collections
import math

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer


class ImprovedSTP(Optimizer):
    """stp maybe better"""
    def __init__(self, params, lr=1e-3, history_size=1000, epsilon=1e-8):
        defaults = dict(lr=lr, history_size=history_size, epsilon=epsilon)
        super().__init__(params, defaults)
        self._state = {}
        self._state['history'] = collections.deque(maxlen=history_size)
        self.lr = lr
        self.history_size = history_size
        self.epsilon = epsilon

    @torch.no_grad
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for ImprovedSTP")

        # Flatten all parameters into a single vector
        params = [p for group in self.param_groups for p in group['params']]
        current_params = parameters_to_vector(params)
        theta = current_params.clone().detach()
        device, dtype = theta.device, theta.dtype

        # Generate perturbation delta using history and random noise
        history = self._state['history']
        m = len(history)
        if m == 0:
            delta = torch.randn_like(theta) * (self.lr ** 0.5)
        else:
            z = torch.randn(m, device=device, dtype=dtype)
            delta_history = sum(z[i] * s for i, s in enumerate(history)) / (m ** 0.5)
            delta_random = torch.randn_like(theta) * (self.lr ** 0.5)
            delta = delta_history + delta_random

        # Evaluate original loss
        with torch.no_grad():
            f0 = closure(False)

        # Evaluate f_plus
        vector_to_parameters(theta + delta, params)
        with torch.no_grad():
            f_plus = closure(False)

        # Evaluate f_minus
        vector_to_parameters(theta - delta, params)
        with torch.no_grad():
            f_minus = closure(False)

        # Restore original parameters
        vector_to_parameters(theta, params)

        # Calculate optimal step from parabola fit
        a = (f_plus + f_minus - 2 * f0) / 2
        b = (f_plus - f_minus) / 2

        if a.abs() < self.epsilon:
            gamma = 0.0
        else:
            gamma = (-b / (2 * a)).item()

        # Evaluate f_parabola if applicable
        if gamma != 0.0:
            vector_to_parameters(theta + gamma * delta, params)
            with torch.no_grad():
                f_parabola = closure(False)
            vector_to_parameters(theta, params)
        else:
            f_parabola = float('inf')

        # Determine best parameters
        candidates = {
            'original': f0,
            'plus': f_plus,
            'minus': f_minus,
            'parabola': f_parabola
        }
        best_key = min(candidates, key=lambda k: candidates[k])

        if best_key == 'original':
            best_theta = theta
            self.lr /= 1.25
        elif best_key == 'plus':
            best_theta = theta + delta
            self.lr *= 1.75
        elif best_key == 'minus':
            best_theta = theta - delta
            self.lr *= 1.75
        else:
            best_theta = theta + gamma * delta

        # Update model parameters
        vector_to_parameters(best_theta, params)

        # Record the step taken if non-zero
        step_taken = best_theta - theta
        if step_taken.norm() > 1e-12:
            if self.history_size > 0: self._state['history'].append(step_taken.detach().clone())

        return candidates[best_key]



class CovarianceSTP(Optimizer):
    def __init__(self, params, lr=1e-3, memory_size=1000,
                 sigma=0.1, diagonal_decay=0.99, epsilon=1e-8):
        defaults = dict(lr=lr, sigma=sigma, diagonal_decay=diagonal_decay, epsilon=epsilon)
        super().__init__(params, defaults)

        # State initialization
        self._state = {}
        self._state['step'] = 0
        self._state['memory'] = collections.deque(maxlen=memory_size)
        self._state['diagonal'] = None  # Will be initialized on first step
        self.sigma = sigma
        self.diagonal_decay = diagonal_decay
        self.epsilon = epsilon

    def _initialize_diagonal(self, params_vector):
        """Initialize diagonal covariance with unit variance"""
        if self._state['diagonal'] is None:
            self._state['diagonal'] = torch.ones_like(params_vector)

    def _update_covariance(self, successful_step):
        # Update diagonal with exponential moving average
        self._state['diagonal'] = (self.diagonal_decay * self._state['diagonal'] +
                                 (1 - self.diagonal_decay) * successful_step**2)

        # Update memory with successful step (normalized)
        self._state['memory'].append(successful_step / (self.sigma + self.epsilon))

    def _generate_perturbation(self, params_vector):
        # Ensure diagonal is initialized
        self._initialize_diagonal(params_vector)

        device, dtype = params_vector.device, params_vector.dtype
        n_params = params_vector.size(0)

        # Start with diagonal covariance
        perturbation = torch.randn_like(params_vector) * torch.sqrt(self._state['diagonal'])

        # Add low-rank terms from memory
        if len(self._state['memory']) > 0:
            # Create matrix from memory vectors
            memory_matrix = torch.stack(list(self._state['memory']), dim=1)

            # Random combination of memory directions
            z = torch.randn(memory_matrix.size(1), device=device, dtype=dtype)
            perturbation += memory_matrix @ z * (self.sigma / math.sqrt(memory_matrix.size(1)))

        # Normalize perturbation magnitude while preserving direction
        perturbation_norm = perturbation.norm()
        if perturbation_norm > self.epsilon:
            perturbation *= self.sigma / perturbation_norm

        return perturbation

    @torch.no_grad
    def step(self, closure):
        self._state['step'] += 1

        # Flatten parameters
        params = [p for group in self.param_groups for p in group['params']]
        current_params = parameters_to_vector(params).detach()
        device, dtype = current_params.device, current_params.dtype

        # Generate structured perturbation
        delta = self._generate_perturbation(current_params)

        # Evaluate three points
        with torch.no_grad():
            # Original loss
            vector_to_parameters(current_params, params)
            f0 = closure(False)

            # Positive perturbation
            vector_to_parameters(current_params + delta, params)
            f_plus = closure(False)

            # Negative perturbation
            vector_to_parameters(current_params - delta, params)
            f_minus = closure(False)

            # Restore parameters
            vector_to_parameters(current_params, params)

        # Fit parabola to determine optimal step
        a = (f_plus + f_minus - 2*f0) / 2
        b = (f_plus - f_minus) / 2

        if abs(a) < self.epsilon:
            gamma = 0.0
        else:
            gamma = (-b/(2*a)).item()

        # Evaluate parabola minimum candidate
        if gamma != 0:
            candidate = current_params + gamma * delta
            vector_to_parameters(candidate, params)
            with torch.no_grad():
                f_parabola = closure(False)
            vector_to_parameters(current_params, params)
        else:
            f_parabola = float('inf')

        # Determine best parameters from all candidates
        candidates = {
            'current': (current_params, f0),
            'plus': (current_params + delta, f_plus),
            'minus': (current_params - delta, f_minus),
            'parabola': (current_params + gamma*delta, f_parabola)
        }
        best_params, best_loss = min(candidates.values(), key=lambda x: x[1])

        # Update model parameters
        vector_to_parameters(best_params, params)

        # Update covariance with successful step (if not original)
        if not torch.allclose(best_params, current_params):
            step_taken = best_params - current_params
            self._update_covariance(step_taken.detach())

        return best_loss