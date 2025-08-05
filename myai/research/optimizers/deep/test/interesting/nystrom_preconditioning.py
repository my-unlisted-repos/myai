import torch
import torch.optim
import torch.optim as optim
import torch.optim.optimizer
import torch.optim.optimizer as optimizer
from torch import optim
from torch.optim import Optimizer, optimizer
from torch.optim.optimizer import Optimizer


class NystromPreconditioning(Optimizer):
    """1. note tested AT ALL; 2. requires closure that returns per-sample loss.

    def closure():
        preds = model(inouts)
        loss = F.mseloss(preds, targets, reduction = 'none').mean((1,2,3))
        opt.zero_grad();
        mean_loss = loss.mean()
        mean_loss.backward()
        return mean_loss, loss

    """
    def __init__(self, params, l=20, lambda_=1e-3, lr=1e-3):
        defaults = dict(l=l, lambda_=lambda_, lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure):
        # Ensure closure returns per-sample losses
        with torch.enable_grad(): loss, per_sample_losses = closure()

        for group in self.param_groups:
            l = group['l']
            lambda_ = group['lambda_']
            lr = group['lr']
            batch_size = per_sample_losses.size(0)

            # Randomly select subset indices
            indices = torch.randperm(batch_size, device=per_sample_losses.device)[:l]
            selected_losses = per_sample_losses[indices]

            for p in group['params']:
                if p.grad is None:
                    continue

                # Compute per-sample gradients for selected samples
                C = []
                for loss_i in selected_losses:
                    # Zero other gradients but retain parameter's graph
                    self.state[p]['temp_grad'] = torch.zeros_like(p)
                    with torch.enable_grad(): grad = torch.autograd.grad(loss_i, p, retain_graph=True)[0]
                    C.append(grad.detach().flatten())

                # Form matrix C (d x l)
                C = torch.stack(C, dim=1)  # Shape: (num_params, l)
                d = C.size(0)

                # Compute W = C^T C + lambda*I for stability
                W = torch.matmul(C.T, C)
                W_reg = W + lambda_ * torch.eye(l, device=W.device)

                # Invert W_reg using Cholesky for numerical stability
                try:
                    U = torch.linalg.cholesky(W_reg)
                    W_reg_inv = torch.cholesky_inverse(U)
                except RuntimeError:
                    # Fallback to pseudo-inverse if Cholesky fails
                    W_reg_inv = torch.pinverse(W_reg)

                # Get average gradient and flatten
                g = p.grad.detach().flatten().unsqueeze(1)  # (d, 1)

                # Compute preconditioned gradient using Woodbury
                C_T_g = torch.matmul(C.T, g)  # (l, 1)
                intermediate = torch.matmul(W_reg_inv, C_T_g)  # (l, 1)
                precond_g = (1/lambda_)*g - (1/lambda_**2)*torch.matmul(C, intermediate)

                # Update parameters
                p.data.add_(-lr * precond_g.reshape(p.data.shape))

                # Cleanup temporary state
                del self.state[p]['temp_grad']

        return loss




class NystromFeaturePreconditioning(optim.Optimizer):
    """not tested at all. STEP TAKES IN HIDDEN FEATURES OF SIZE [batch_size, feature_dimension]"""
    def __init__(self, params, lr=1e-3, nystrom_rank=50, momentum=0, dampening=0, weight_decay=0):
        if not isinstance(params, (list, tuple)) or not all(isinstance(p, torch.nn.Parameter) for p in params):
            raise TypeError("params must be a list or tuple of Parameters")

        defaults = dict(lr=lr, nystrom_rank=nystrom_rank, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.approx_inv_hessian = None  # Store the approximate inverse Hessian
        self.prev_batch_features = None # Store features from the previous batch
        self.momentum_buffer = None # Momentum buffer

    @torch.no_grad
    def step(self, closure=None, features=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                  state['step'] = 0
                  if group['momentum'] > 0:
                      state['momentum_buffer'] = torch.zeros_like(p.data)

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Nystrom Preconditioning
                if features is not None: # only if features are provided
                    current_batch_features = features

                    if self.prev_batch_features is not None:  # Use previous batch for low-rank approx
                        # Construct the data matrix for Nystrom approximation (combine previous and current).
                        # This could be improved for memory efficiency by using a sliding window.
                        combined_features = torch.cat([self.prev_batch_features, current_batch_features], dim=0)

                        # Efficiently compute the kernel matrix and its eigendecomposition.
                        # Using a simple dot product kernel here. More complex kernels are possible.
                        kernel_matrix = torch.mm(combined_features, combined_features.t())

                        # Eigendecomposition (you might want to use a more stable method for large matrices).
                        eigenvalues, eigenvectors = torch.linalg.eigh(kernel_matrix)

                        # Select top-k eigenvalues and eigenvectors for the Nyström approximation.
                        top_k_indices = torch.argsort(eigenvalues, descending=True)[:group['nystrom_rank']]
                        eigenvalues_k = eigenvalues[top_k_indices]
                        eigenvectors_k = eigenvectors[:, top_k_indices]

                        # Construct the approximate inverse Hessian.  Adding small value for numerical stability
                        approx_inv_hessian = torch.mm(eigenvectors_k, torch.mm(torch.diag(1.0 / (eigenvalues_k+1e-8)), eigenvectors_k.t()))

                        self.approx_inv_hessian = approx_inv_hessian # Store for the next iteration
                    else:
                        # Initialize for first step
                        kernel_matrix = torch.mm(current_batch_features, current_batch_features.t())
                        eigenvalues, eigenvectors = torch.linalg.eigh(kernel_matrix)
                        top_k_indices = torch.argsort(eigenvalues, descending=True)[:group['nystrom_rank']]
                        eigenvalues_k = eigenvalues[top_k_indices]
                        eigenvectors_k = eigenvectors[:, top_k_indices]
                        approx_inv_hessian = torch.mm(eigenvectors_k, torch.mm(torch.diag(1.0 / (eigenvalues_k+1e-8)), eigenvectors_k.t()))
                        self.approx_inv_hessian = approx_inv_hessian

                    self.prev_batch_features = current_batch_features # Store current features for the next iteration

                    # Preconditioned gradient. Reshape is very important
                    preconditioned_grad = torch.mm(grad.view(1,-1), self.approx_inv_hessian).view(grad.shape)

                    if group['momentum'] > 0:
                        momentum_buffer = state['momentum_buffer']
                        momentum_buffer.mul_(group['momentum']).add_(preconditioned_grad)
                        preconditioned_grad = momentum_buffer

                    p.data.add_(preconditioned_grad, alpha=-group['lr'])

                else: # Fallback to standard gradient descent if features are not provided
                    if group['momentum'] > 0:
                        momentum_buffer = state['momentum_buffer']
                        momentum_buffer.mul_(group['momentum']).add_(grad)
                        d_p = momentum_buffer
                    else:
                        d_p = grad

                    p.data.add_(d_p, alpha=-group['lr'])

        return loss