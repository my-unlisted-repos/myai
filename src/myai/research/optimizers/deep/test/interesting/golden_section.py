import torch
from torch.optim import Optimizer

class AdaptiveGGGSSOptimizer(Optimizer):
    """
    Adaptive Gradient-Guided Golden-Section Search (Adaptive-GG-GSS) Optimizer.

    Refinement: Adapts search direction by combining gradients evaluated during
    Golden-section search within each step. Aims to be more genuinely multivariate
    than simple line search along a fixed gradient direction.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Initial interval size for Golden-section search.
        golden_section_iterations (int, optional): Number of Golden-section iterations.
        phi (float, optional): Golden ratio.
        direction_adaptation (bool, optional): Whether to adapt search direction
            based on gradients during Golden-section search (default: True).
    """

    def __init__(self, params, lr=1e-3, golden_section_iterations=10, phi=(1 + 5**0.5) / 2, direction_adaptation=True):


        defaults = dict(lr=lr, golden_section_iterations=golden_section_iterations, phi=phi, direction_adaptation=direction_adaptation)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step."""

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['lower_bound'] = -group['lr']
                    state['upper_bound'] = group['lr']
                    state['search_direction'] = -grad.flatten() # Initial direction: negative gradient

                lower_bound = state['lower_bound']
                upper_bound = state['upper_bound']
                phi = group['phi']
                iterations = group['golden_section_iterations']
                direction_adaptation_enabled = group['direction_adaptation']

                original_p = p.data.clone()
                current_direction = state['search_direction']
                direction_norm = torch.linalg.norm(current_direction)

                if direction_norm == 0: # No direction, skip update
                    continue
                current_direction = current_direction / direction_norm

                # Lists to store gradients and losses during GSS for direction adaptation
                gss_gradients = []
                gss_losses = []

                for _ in range(iterations):
                    a = lower_bound
                    b = upper_bound

                    x1 = b - (b - a) / phi
                    x2 = a + (b - a) / phi

                    param_x1 = original_p + (x1 * current_direction).reshape(p.shape)
                    param_x2 = original_p + (x2 * current_direction).reshape(p.shape)

                    def evaluate_and_get_grad(params_val):
                        p.data = params_val
                        with torch.enable_grad():
                            loss_val = closure() # Call closure to get loss AND compute gradients
                        grad_val = p.grad.data.clone().flatten() # Get gradient at this point
                        return loss_val, grad_val

                    loss_x1, grad_x1 = evaluate_and_get_grad(param_x1)
                    loss_x2, grad_x2 = evaluate_and_get_grad(param_x2)

                    gss_losses.append(loss_x1.item()) # Store loss values (for potential weighting later)
                    gss_losses.append(loss_x2.item())
                    gss_gradients.append(grad_x1) # Store gradients at x1 and x2
                    gss_gradients.append(grad_x2)


                    if loss_x1 < loss_x2:
                        upper_bound = x2
                    else:
                        lower_bound = x1

                best_step = (lower_bound + upper_bound) / 2.0
                p.data = original_p - (best_step * current_direction).reshape(p.shape)


                if direction_adaptation_enabled and gss_gradients: # Adapt direction if enabled and gradients collected
                    # Simple direction adaptation: Average of gradients at x1, x2 and current gradient (grad)
                    # More sophisticated could be weighted average based on losses (or other heuristics)
                    adapted_direction = torch.zeros_like(grad).flatten()
                    adapted_direction += grad.flatten() # Include current gradient
                    for g in gss_gradients: # Include gradients from GSS evaluations
                        adapted_direction += g

                    if torch.linalg.norm(adapted_direction) > 0: # Avoid division by zero
                        state['search_direction'] = adapted_direction # Update search direction for next step
                    else:
                        state['search_direction'] = -grad.flatten() # Fallback to -grad if no meaningful direction


                state['lower_bound'] = lower_bound
                state['upper_bound'] = upper_bound


        return loss