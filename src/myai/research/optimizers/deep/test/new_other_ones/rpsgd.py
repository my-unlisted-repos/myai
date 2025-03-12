import torch
from torch.optim import Optimizer

class RPSGD(Optimizer):
    """
    Randomized Preconditioned Stochastic Gradient Descent (RPSGD) optimizer.
    
    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate. Default: 1e-3.
        beta (float, optional): Exponential decay rate for the diagonal preconditioner. Default: 0.9.
        k (int, optional): Number of random buckets for Count-Sketch. Default: 5.
        eps (float, optional): Term added to denominator to improve numerical stability. Default: 1e-8.
    """
    
    def __init__(self, params, lr=1e-3, beta=0.9, k=5, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, k=k, eps=eps)
        super(RPSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['D'] = torch.ones_like(p.data)
                
                beta = group['beta']
                lr = group['lr']
                k = group['k']
                eps = group['eps']
                
                # Update diagonal preconditioner D
                state['D'].mul_(beta).addcmul_(1 - beta, grad, grad)
                
                # Flatten gradient and D
                flat_grad = grad.contiguous().view(-1)
                D = state['D'].contiguous().view(-1)
                inv_D = 1.0 / (D.sqrt() + eps)
                n = flat_grad.size(0)
                
                # Generate random hash functions h (buckets) and s (signs)
                h = torch.randint(0, k, (n,), device=grad.device)
                s = torch.randint(0, 2, (n,), device=grad.device, dtype=torch.float32) * 2 - 1  # -1 or 1
                
                # Compute S^T * (inv_D * grad) via scatter_add
                ST_g = torch.zeros(k, device=grad.device)
                ST_g.scatter_add_(0, h, s * inv_D * flat_grad)
                
                # Compute diagonal of S^T * inv_D * S (sum inv_D for each bucket)
                ST_inv_D_S_diag = torch.zeros(k, device=grad.device)
                ST_inv_D_S_diag.scatter_add_(0, h, inv_D)
                
                # Compute Minv = (I + ST_inv_D_S_diag)^{-1} as diagonal
                Minv_diag = 1.0 / (1 + ST_inv_D_S_diag + eps)
                
                # Compute correction term: inv_D * S * Minv_diag * ST_g
                # Gather Minv_diag and ST_g for each index
                Minv_diag_h = Minv_diag[h]
                ST_g_h = ST_g[h]
                correction = s * Minv_diag_h * ST_g_h
                correction.mul_(inv_D)
                
                # Preconditioned gradient
                precond_grad_flat = inv_D * flat_grad - correction
                precond_grad = precond_grad_flat.view_as(p.data)
                
                # Update parameters
                p.data.add_(-lr, precond_grad)
                
                state['step'] += 1
        
        return loss