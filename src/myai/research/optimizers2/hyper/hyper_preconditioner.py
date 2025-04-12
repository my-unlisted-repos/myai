import torch
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy

class HyperLowRankOptimizer(optim.Optimizer):
    """
    Implements Hypergradient Descent for a Low-Rank Preconditioner.

    Updates parameters `w` using a preconditioned gradient:
        w_{t+1} = w_t - lr * P_t * grad(L(w_t))
    where P_t = U_t @ V_t.T (or U_t @ U_t.T if symmetric=True).

    Updates the low-rank factors U_t, V_t using hypergradient descent:
        U_{t+1} = U_t - beta * grad_U H_t
        V_{t+1} = V_t - beta * grad_V H_t
    where H_t = L(w_{t+1}) and gradients are approximated using one-step info.

    Args:
        params (iterable): Iterable of parameters to optimize (model parameters).
        lr (float): Learning rate for parameter updates (alpha in derivation).
        rank (int): Rank 'k' of the preconditioner.
        beta (float): Learning rate for hyperparameter (U, V) updates.
        symmetric (bool): If True, use P = U @ U.T (V is ignored). Default: False.
        epsilon (float): Small value for diagonal loading (stability). P_eff = eps*I + P. Default: 1e-8.
        hyper_optimizer_type (str): 'adam' or 'sgd' for updating U, V. Default: 'adam'.
        hyper_optimizer_options (dict): Options for the hyper optimizer (e.g., betas for Adam).
        device (torch.device): Device for tensors. If None, inferred from params.
    """
    def __init__(self, params, lr=1e-3, rank=10, beta=1e-2,
                 symmetric=False, epsilon=1e-8,
                 hyper_optimizer_type='adam',
                 hyper_optimizer_options=None,
                 device=None):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if beta < 0.0:
            raise ValueError(f"Invalid hyper learning rate: {beta}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")

        # Extract parameters and determine total size and device
        if isinstance(params, torch.Tensor):
            params = [params]
        self.param_list = list(params) # Ensure it's a list
        if not self.param_list:
            raise ValueError("Optimizer received an empty parameter list")

        # Determine device and total dimension
        if device is None:
            self.device = self.param_list[0].device
        else:
            self.device = device

        # Calculate total dimension d by flattening parameters
        with torch.no_grad():
            dummy_vector = parameters_to_vector(self.param_list)
        d = dummy_vector.numel()
        del dummy_vector

        defaults = dict(lr=lr, rank=rank, beta=beta, symmetric=symmetric,
                        epsilon=epsilon, d=d,
                        hyper_optimizer_type=hyper_optimizer_type)
        super().__init__(self.param_list, defaults)

        # Initialize U and V factors
        # Initialize near zero to start close to standard SGD (+epsilon*I)
        stdv = 1. / (rank * d)**0.5 # Small random init
        self.U = torch.nn.Parameter(torch.randn(d, rank, device=self.device) * stdv)
        if not symmetric:
            self.V = torch.nn.Parameter(torch.randn(d, rank, device=self.device) * stdv)
        else:
            self.V = None # V is not used in symmetric mode

        # State to store previous gradient g_t
        self.state['prev_grad'] = None

        # Setup hyperparameter optimizer
        hyper_params = [self.U]
        if not symmetric:
            hyper_params.append(self.V)

        hyper_opt_options = hyper_optimizer_options if hyper_optimizer_options is not None else {}
        if hyper_optimizer_type.lower() == 'adam':
            self.hyper_optimizer = optim.Adam(hyper_params, lr=beta, **hyper_opt_options)
        elif hyper_optimizer_type.lower() == 'sgd':
            self.hyper_optimizer = optim.SGD(hyper_params, lr=beta, **hyper_opt_options)
        else:
            raise ValueError(f"Unsupported hyper_optimizer_type: {hyper_optimizer_type}")

        # Store parameter shapes for unflattening
        self._param_shapes = [p.shape for p in self.param_list]
        self._param_numels = [p.numel() for p in self.param_list]


    @torch.no_grad()
    def _flatten_grads(self):
        """Flattens gradients from self.param_list into a single vector."""
        grads = []
        for p in self.param_list:
            if p.grad is None:
                grads.append(torch.zeros_like(p).view(-1))
            else:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)

    @torch.no_grad()
    def _unflatten_vector(self, vector):
        """Converts a flat vector back to a list of tensors with original shapes."""
        return vector.split(self._param_numels)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                                and returns the loss. This is needed to
                                compute g_{t+1}.
        """
        group = self.param_groups[0] # Assuming one param group
        lr = group['lr']
        beta = group['beta']
        rank = group['rank']
        symmetric = group['symmetric']
        epsilon = group['epsilon']
        d = group['d']

        # --- Step (a) & Store g_t ---
        # We assume g_t was computed *before* calling step (e.g., loss.backward())
        # Or typically, the closure computes it *first*. Let's rely on closure.

        # Zero grads before computing g_t using the closure
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure() # This should compute gradients for w_t

        g_t_flat = self._flatten_grads()

        # Store g_t if it's the first hypergradient step
        if self.state['prev_grad'] is None:
             # On the very first step, we don't have g_t and g_{t+1} yet.
             # We can either skip the hypergradient update or use g_t as g_{t+1}.
             # Skipping is safer. Let's just perform a standard step for t=0.
             # For simplicity here, we'll require g_t to be available from a previous step for hyperupdate.
             # A more robust implementation might handle the t=0 case explicitly.
             # For now, let's assume g_{t-1} is stored as prev_grad, and g_t is current.
             # Let's rename: g_prev = self.state['prev_grad'], g_curr = g_t_flat
             g_prev = self.state['prev_grad']
             g_curr = g_t_flat.clone() # Store current grad for next iteration
             self.state['prev_grad'] = g_curr
             has_hypergradient_info = g_prev is not None
        else:
             g_prev = self.state['prev_grad'] # This is g_t from the derivation
             g_curr = g_t_flat.clone()      # This is g_{t+1} from the derivation
             self.state['prev_grad'] = g_curr # Store for next step
             has_hypergradient_info = True


        # --- Step (b) Precondition Gradient ---
        # g_p = (eps * I + U V^T) g_curr  OR g_p = (eps * I + U U^T) g_curr
        g_t_for_precond = g_curr # Use current gradient for parameter update

        if symmetric:
            # P_g = U @ (U.t() @ g_t_for_precond)
            intermediate = self.U.t() @ g_t_for_precond
            preconditioned_part = self.U @ intermediate
        else:
            # P_g = U @ (V.t() @ g_t_for_precond)
            intermediate = self.V.t() @ g_t_for_precond
            preconditioned_part = self.U @ intermediate

        g_p_flat = epsilon * g_t_for_precond + preconditioned_part

        # --- Step (c) Update Parameters w_t -> w_{t+1} ---
        # Unflatten g_p_flat and apply the update manually
        update_vectors = self._unflatten_vector(g_p_flat)
        for p, update_vec in zip(self.param_list, update_vectors):
            p.data.add_(update_vec.view_as(p), alpha=-lr)
            # Ensure gradient is cleared after update, ready for next closure call if needed
            if p.grad is not None:
                 p.grad = None


        # --- Step (e) Calculate Hypergradients (using g_prev and g_curr) ---
        # Note: In our notation g_prev=g_t, g_curr=g_{t+1} from the derivation.
        if has_hypergradient_info:
            self.hyper_optimizer.zero_grad() # Zero U, V grads

            if symmetric:
                # ∇_U H ≈ -α (g_t (g_{t+1}^T U) + g_{t+1} (g_t^T U))
                # ∇_U H ≈ -lr * (g_prev @ (g_curr.t() @ self.U) + g_curr @ (g_prev.t() @ self.U))
                term1_scalar = g_curr.t() @ self.U # (1 x k)
                term1 = torch.outer(g_prev, term1_scalar.squeeze()) # (d x 1) @ (1 x k) -> (d x k)
                term2_scalar = g_prev.t() @ self.U # (1 x k)
                term2 = torch.outer(g_curr, term2_scalar.squeeze()) # (d x 1) @ (1 x k) -> (d x k)
                grad_U_H = -lr * (term1 + term2)
                if self.U.grad is None:
                     self.U.grad = grad_U_H
                else:
                     self.U.grad.copy_(grad_U_H) # Use copy_ for efficiency
            else:
                # ∇_U H ≈ -α g_{t+1} (g_t^T V) = -lr * g_curr @ (g_prev.t() @ self.V)
                term_u_scalar = g_prev.t() @ self.V # (1 x k)
                grad_U_H = -lr * torch.outer(g_curr, term_u_scalar.squeeze()) # (d x k)

                # ∇_V H ≈ -α g_t (g_{t+1}^T U) = -lr * g_prev @ (g_curr.t() @ self.U)
                term_v_scalar = g_curr.t() @ self.U # (1 x k)
                grad_V_H = -lr * torch.outer(g_prev, term_v_scalar.squeeze()) # (d x k)

                if self.U.grad is None:
                    self.U.grad = grad_U_H
                else:
                    self.U.grad.copy_(grad_U_H)
                if self.V.grad is None:
                    self.V.grad = grad_V_H
                else:
                    self.V.grad.copy_(grad_V_H)

            # --- Step (f) Update Preconditioner Factors ---
            self.hyper_optimizer.step()

            # Optional: Add stabilization like clipping norms (not implemented here)
            # with torch.no_grad():
            #     torch.nn.utils.clip_grad_norm_(self.U, max_norm=1.0)
            #     if not symmetric:
            #         torch.nn.utils.clip_grad_norm_(self.V, max_norm=1.0)

        # Return the loss computed at the beginning of the step (loss_t)
        return loss


    def zero_hyper_grad(self):
        """Clears the gradients of the hyperparameters U (and V)."""
        self.hyper_optimizer.zero_grad()

    def zero_grad(self, set_to_none=True):
        """Clears the gradients of the model parameters."""
        super().zero_grad(set_to_none=set_to_none)
