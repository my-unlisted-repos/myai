import torch
from torch.optim import Optimizer

class AdaptGrad(Optimizer):
    """
    very basic idea of adding difference between last gradients, need to test both postiive and negative adapt factors.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0)
        adapt_factor (float, optional): factor for gradient adaptation (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        dampening (float, optional): dampening for momentum (default: 0)
        maximize (bool, optional): maximize the objective function (default: False)
    """

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, adapt_factor=1., maximize=False):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        adapt_factor=adapt_factor, maximize=maximize)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            nesterov = group['nesterov']
            if isinstance(nesterov, bool):
                group['nesterov'] = nesterov

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            adapt_factor = group['adapt_factor']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if maximize:
                    d_p = -d_p
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                param_state = self.state[p]
                momentum_buffer = param_state['momentum_buffer']
                previous_grad = param_state['previous_grad']

                # Compute gradient difference
                grad_diff = d_p - previous_grad
                param_state['previous_grad'] = d_p.clone() # Store current grad for next step

                if momentum != 0:
                    buf = momentum_buffer
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    # momentum_buffer_np1 = buf.clone() # Store for adapt_term calculation
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # Adaptation term based on gradient difference
                if adapt_factor != 0:
                    adapt_term = grad_diff * adapt_factor # adaptation term
                    d_p.add_(adapt_term)


                p.add_(d_p, alpha=-group['lr'])

        return loss


