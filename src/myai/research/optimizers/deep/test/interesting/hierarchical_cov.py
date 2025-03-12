import torch
import torch.nn as nn
import torch.optim as optim


class TreeNode:
    """Represents a node in the parameter tree."""
    def __init__(self, params):
        self.params = params  # List of parameters associated with this node

class HierarchicalCovariance(optim.Optimizer):
    def __init__(self, params, rank=5, solver_iterations=3, lr=0.01):
        params = list(params)
        defaults = dict(lr=lr, rank=rank, solver_iterations=solver_iterations)
        super().__init__(params, defaults)
        #self.tree_structure = tree_structure  # List of TreeNode objects
        self.tree_structure = [TreeNode([p]) for p in params]
        self.rank = rank
        self.solver_iterations = solver_iterations
        self._initialize_decompositions()

    def _initialize_decompositions(self):
        """Initialize low-rank approximations for each node."""
        for node in self.tree_structure:
            param_subset = self._get_params_for_node(node)
            num_params = sum(p.numel() for p in param_subset)
            device = param_subset[0].device
            self.state[node] = {
                'U': torch.randn(num_params, self.rank, device=device),
                'S': torch.ones(self.rank, device=device),  # Singular values
                'grad_buffer': []  # Buffer for gradient updates
            }
            # Normalize U to ensure stability
            if num_params >= self.rank: self.state[node]['U'], _ = torch.qr(self.state[node]['U'])


    def _get_params_for_node(self, node):
        """Retrieve parameters for a given node."""
        return node.params

    def step(self, closure=None):
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                preconditioned_grad = self._compute_preconditioned_grad(p, grad)
                p.data.add_(-group['lr'], preconditioned_grad)

        self._update_decompositions()
        return loss

    def _compute_preconditioned_grad(self, param, grad):
        """Compute preconditioned gradient using hierarchical approximation."""
        node = self._find_node_for_param(param)
        if not node or node not in self.state:
            return grad  # Fallback to vanilla gradient

        state = self.state[node]
        U, S = state['U'], state['S']
        grad_flat = grad.view(-1)

        # Define matrix-vector product: H v ≈ U (S * (U^T v))
        def matvec(v):
            return U @ (S * (U.T @ v))

        # Approximate inverse Hessian-vector product using conjugate gradient
        # For simplicity, use a projection here; in practice, use torch.linalg.lstsq or CG
        coeffs = U.T @ grad_flat
        preconditioned = U @ (coeffs / (S + 1e-8))  # Regularize S to avoid division by zero
        return preconditioned.view_as(grad)

    def _find_node_for_param(self, param):
        """Find the tree node containing the parameter."""
        for node in self.tree_structure:
            if id(param) in [id(p) for p in node.params]:
                return node
        return None

    def _update_decompositions(self):
        """Update low-rank approximations using online PCA."""
        for node in self.tree_structure:
            param_subset = self._get_params_for_node(node)
            grads = [p.grad.data for p in param_subset if p.grad is not None]
            if not grads:
                continue
            grad_vec = torch.cat([g.view(-1) for g in grads])
            state = self.state[node]
            U, S = state['U'], state['S']

            # Online PCA: Project and update
            proj = U.T @ grad_vec
            residual = grad_vec - U @ proj
            residual_norm = torch.norm(residual)
            if residual_norm > 1e-6:  # Only update if residual is significant
                U_new = torch.cat([U, residual.unsqueeze(1) / residual_norm], dim=1)
                S_new = torch.cat([S, residual_norm.unsqueeze(0)])
                # Truncate to rank
                if U_new.shape[1] > self.rank:
                    U_new, S_new, _ = torch.svd_lowrank(U_new * S_new, q=self.rank)
                    state['U'] = U_new
                    state['S'] = S_new[:self.rank]
                else:
                    state['U'] = U_new
                    state['S'] = S_new
