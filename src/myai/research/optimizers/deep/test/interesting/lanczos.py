import torch
from torch.optim import Optimizer

class LanczosOpt(Optimizer):
    """
    Implements Lanczos-based optimization.

    Algorithm inspired by using Lanczos iteration to approximate Hessian action
    for efficient stochastic optimization. This is a novel adaptation.
    """

    def __init__(self, params, lr=1e-3, lanczos_steps=5, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, beta1=0.9, beta2=0.999, eps=1e-8,
                 use_adam_components=False):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, lanczos_steps=lanczos_steps,
                        beta1=beta1, beta2=beta2, eps=eps, use_adam_components=use_adam_components)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            lanczos_steps = group['lanczos_steps']
            beta1, beta2 = group['beta1'], group['beta2']
            eps = group['eps']
            use_adam_components = group['use_adam_components']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                param_state = self.state[p]

                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p)
                if use_adam_components and 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p)
                    param_state['exp_avg_sq'] = torch.zeros_like(p)
                    param_state['step'] = 0

                if momentum != 0:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if use_adam_components:
                    param_state['step'] += 1
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt()).add_(eps)
                    bias_correction1 = 1 - beta1 ** param_state['step']
                    bias_correction2 = 1 - beta2 ** param_state['step']
                    step_size = group['lr'] * torch.sqrt(bias_correction2) / bias_correction1
                    d_p = exp_avg / denom * step_size # Use Adam-style direction


                # Lanczos iteration to refine direction - Simple version: Gradient Krylov
                q = d_p.reshape(-1) # Start with gradient vector
                alpha_coeffs = []
                beta_coeffs = []
                Q = [] # Lanczos vectors
                Q.append(q / torch.linalg.norm(q) if torch.linalg.norm(q) > 0 else torch.zeros_like(q)) # Normalize q_0
                if torch.linalg.norm(q) <= 0:
                    update_direction = torch.zeros_like(d_p) # Zero gradient case
                else:
                    q = Q[0]
                    for k in range(lanczos_steps):
                        matrix_vector_product = q # Simple approximation: Identity-like operation. Could be replaced by Hessian-vector product approximation
                        alpha_k = torch.dot(matrix_vector_product, Q[-1])
                        alpha_coeffs.append(alpha_k)
                        q = matrix_vector_product - alpha_k * Q[-1]
                        if k > 0:
                            q = q - beta_coeffs[-1] * Q[-2]

                        beta_k = torch.linalg.norm(q)
                        beta_coeffs.append(beta_k)
                        if beta_k != 0 and k < lanczos_steps -1:
                            Q.append(q / beta_k)
                        else:
                            Q.append(torch.zeros_like(q)) # In case beta_k is zero, or last iteration


                    # Construct update direction from Lanczos basis and coeffs (simple projection for now)
                    update_direction_flat = torch.zeros_like(d_p.reshape(-1))
                    if Q[0].abs().sum() > 0: # Avoid projection if initial Q is zero vector
                        for i in range(min(lanczos_steps, len(Q))):
                            if Q[i].abs().sum() > 0: # Only project onto non-zero basis vectors
                                update_direction_flat = update_direction_flat + torch.dot(d_p.reshape(-1), Q[i]) * Q[i]

                    update_direction = update_direction_flat.reshape(d_p.shape)


                p.add_(update_direction, alpha=-group['lr']) # Apply Lanczos refined direction

        return loss