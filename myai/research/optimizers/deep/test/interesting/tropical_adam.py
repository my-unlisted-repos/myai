import torch
from torch.optim import Optimizer

class TropicalAdam(Optimizer): # grok 3 version
    """
    Implements a tropical geometry version of the Adam optimizer in PyTorch.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        a (float, optional): Decay factor for the first moment (default: 0.1).
        b (float, optional): Decay factor for the second moment (default: 0.1).
        eps (float, optional): Small constant to prevent division by zero (default: 1e-8).
    """
    def __init__(self, params, lr=1e-3, a=0.1, b=0.1, eps=1e-8):
        # Define default hyperparameters
        defaults = dict(lr=lr, a=a, b=b, eps=eps)
        # Initialize the parent Optimizer class
        super(TropicalAdam, self).__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        # Iterate over parameter groups
        for group in self.param_groups:
            for p in group['params']:
                # Skip if the parameter has no gradient
                if p.grad is None:
                    continue

                # Get the gradient tensor
                grad = p.grad.data

                # Get or initialize the state for this parameter
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m_plus'] = torch.zeros_like(p.data)
                    state['m_minus'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                # Increment step count
                state['step'] += 1

                # Retrieve hyperparameters
                a = group['a']
                b = group['b']
                lr = group['lr']
                eps = group['eps']

                # Get current state values
                m_plus = state['m_plus']
                m_minus = state['m_minus']
                v = state['v']

                # Tropical update for m_plus (positive gradients)
                m_plus_new = torch.min(m_plus + a, torch.max(grad, torch.zeros_like(grad)))

                # Tropical update for m_minus (negative gradients)
                m_minus_new = torch.min(m_minus + a, torch.max(-grad, torch.zeros_like(grad)))

                # Compute effective first moment
                m_t = m_plus_new - m_minus_new

                # Tropical update for second moment (magnitude)
                v_new = torch.min(v + b, torch.abs(grad))

                # Parameter update
                p.data = p.data - lr * m_t / (v_new + eps)

                # Save updated state
                state['m_plus'] = m_plus_new
                state['m_minus'] = m_minus_new
                state['v'] = v_new
