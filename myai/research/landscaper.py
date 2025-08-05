# pylint: disable=redefined-outer-name
import copy
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm


# Helper functions for flattening/unflattening parameters/gradients
def _flatten_params(params):
    """Flattens a list of tensors into a single 1D tensor."""
    vecs = [p.data.view(-1) for p in params]
    if not vecs:
        return torch.tensor([])
    return torch.cat(vecs)

def _flatten_grads(params):
    """Flattens gradients corresponding to a list of parameters."""
    vecs = []
    for p in params:
        if p.grad is None:
            # Handle cases where some params might not have grads (e.g., frozen layers)
            vecs.append(torch.zeros_like(p.data).view(-1))
        else:
            vecs.append(p.grad.data.view(-1))
    if not vecs:
        return torch.tensor([])
    return torch.cat(vecs)

def _unflatten_params(vec, params_like):
    """Unflattens a 1D tensor back into a list of tensors with shapes like params_like."""
    idx = 0
    params_out = []
    for p_ref in params_like:
        numel = p_ref.data.numel()
        if numel > 0: # Skip zero-element tensors if any
            params_out.append(vec[idx:idx + numel].view_as(p_ref.data).clone())
            idx += numel
        else:
            params_out.append(p_ref.data.clone()) # Keep the empty tensor structure
    # Safety check
    if idx != vec.numel():
        print(f"Warning: Parameter shapes mismatch during unflattening. Used {idx}, expected {vec.numel()}.")
        # Fallback: try unflattening by cumulative elements, might handle edge cases better
        idx = 0
        params_out = []
        cumulative_elements = [0] + list(np.cumsum([p.data.numel() for p in params_like]))
        for i, p_ref in enumerate(params_like):
            start, end = cumulative_elements[i], cumulative_elements[i+1]
            params_out.append(vec[start:end].view_as(p_ref.data).clone())
        if cumulative_elements[-1] != vec.numel():
            raise ValueError("Irrecoverable parameter shapes mismatch during unflattening.")


    return params_out

def _set_params(model_params_list, new_params_list):
    """Sets model parameters from a list of tensors."""
    with torch.no_grad():
        for p_model, p_new in zip(model_params_list, new_params_list):
            p_model.data.copy_(p_new.data)

class VisualizingOptimizer(Optimizer):
    def __init__(self, params, base_optimizer_cls=optim.Adam, n_steps_opt=50,
                 subspace_method='active_subspace', # 'active_subspace' or 'pca_params'
                 grid_resolution=20, ars_samples=100, range_extension_factor=1.5,
                 max_range_extensions=3, viz_batch_size=None, device='cpu',
                 **base_optimizer_kwargs):
        """
        Args:
            params: Iterable of parameters to optimize.
            base_optimizer_cls: The underlying optimizer class (e.g., optim.Adam, optim.SGD).
            n_steps_opt: Number of standard optimization steps per cycle.
            subspace_method: 'active_subspace' or 'pca_params'.
            grid_resolution: Number of points along each axis for the grid search.
            ars_samples: Number of samples for Accelerated Random Search.
            range_extension_factor: Factor to extend the initial grid range based on trajectory.
            max_range_extensions: Max times to extend the grid if best point is on edge.
            viz_batch_size: Batch size for evaluating loss during visualization (optional, for memory).
            device: Device for calculations ('cpu' or 'cuda').
            **base_optimizer_kwargs: Keyword arguments for the base_optimizer_cls (e.g., lr=0.001).
        """
        if n_steps_opt <= 1:
            raise ValueError("n_steps_opt must be > 1 to define a subspace.")
        if subspace_method not in ['active_subspace', 'pca_params']:
            raise ValueError("subspace_method must be 'active_subspace' or 'pca_params'")

        # Create a dummy default dict first, then initialize the base optimizer properly
        defaults = dict()
        super().__init__(params, defaults)

        # Initialize the base optimizer *after* super().__init__ call
        # We need self.param_groups to be populated
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups # Use groups from base optimizer

        # Store config
        self.n_steps_opt = n_steps_opt
        self.subspace_method = subspace_method
        self.grid_resolution = grid_resolution
        self.ars_samples = ars_samples
        self.range_extension_factor = range_extension_factor
        self.max_range_extensions = max_range_extensions
        self.viz_batch_size = viz_batch_size
        self.device = torch.device(device)

        # State variables
        self.opt_step_count = 0
        self.param_history = []
        self.grad_history = []
        self.loss_history = []
        self.visualization_frames = [] # Stores PIL Images
        self._model_params_list_ref = list(self.param_groups[0]['params']) # Store a reference list


    def zero_grad(self, set_to_none: bool = False):
        # Delegate zero_grad to the base optimizer
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _get_active_subspace_directions(self, n_components=2):
        """Computes dominant directions using Active Subspaces."""
        if not self.grad_history:
            print("Warning: No gradient history for Active Subspace. Falling back to PCA on params.")
            return self._get_pca_directions(n_components)

        #print("Calculating Active Subspace directions...")
        # Ensure gradients are on the correct device and float
        flat_grads = torch.stack([g.to(self.device).float() for g in self.grad_history])
        num_grads, dim = flat_grads.shape

        if dim == 0:
            raise ValueError("Cannot compute Active Subspace with 0-dimensional parameters.")

        # Compute covariance matrix C = (1/N) * sum(g_i * g_i^T)
        # Efficient computation: C = (1/N) * G^T @ G
        # However, G might be large (num_grads x dim). If dim is large, computing G^T @ G is better.
        # If num_grads is much smaller than dim, compute G @ G^T (num_grads x num_grads),
        # find its eigenvectors 'u', then principal directions are G^T @ u.
        if num_grads < dim:
            #print(f"Using N={num_grads} < D={dim} trick for AS Covariance.")
            cov_small = (1.0 / num_grads) * (flat_grads @ flat_grads.T)
            eigvals, eigvecs_small = torch.linalg.eigh(cov_small) # eigvecs_small are N x N # pylint:disable=not-callable
            # Sort eigenvalues/vectors in descending order
            sorted_indices = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[sorted_indices]
            eigvecs_small = eigvecs_small[:, sorted_indices]
            # Map back to original space D x N @ N x k = D x k
            eigvecs = flat_grads.T @ eigvecs_small
            # Normalize the resulting eigenvectors in the D-dimensional space
            eigvecs = torch.nn.functional.normalize(eigvecs, dim=0)

        else:
            #print(f"Using standard D={dim} x D={dim} Covariance for AS.")
            covariance_matrix = (1.0 / num_grads) * (flat_grads.T @ flat_grads)
            # Eigen decomposition
            try:
                eigvals, eigvecs = torch.linalg.eigh(covariance_matrix) # pylint:disable=not-callable
                 # Sort eigenvalues/vectors in descending order
                sorted_indices = torch.argsort(eigvals, descending=True)
                eigvals = eigvals[sorted_indices]
                eigvecs = eigvecs[:, sorted_indices]
            except torch._C._LinAlgError as e: # type:ignore
                print(f"Warning: Eigendecomposition failed for Active Subspace ({e}). Falling back to PCA on params.")
                return self._get_pca_directions(n_components)


        # Select top n_components eigenvectors
        directions = [eigvecs[:, i].clone() for i in range(n_components)]

        # Orthogonalize (should be orthogonal from eigh, but enforce for numerical stability)
        if n_components > 1:
            directions[1] = directions[1] - torch.dot(directions[1], directions[0]) * directions[0]
            directions[1] = torch.nn.functional.normalize(directions[1], dim=0)
            # Add more orthogonalization steps if n_components > 2

        #print("Active Subspace directions computed.")
        return directions

    def _get_pca_directions(self, n_components=2):
        """Computes dominant directions using PCA on parameter history."""
        #print("Calculating PCA directions on parameters...")
        if len(self.param_history) <= 1:
            raise ValueError("Need at least 2 points in history for PCA.")

        flat_params = torch.stack([p.to(self.device).float() for p in self.param_history]) # N x D
        if flat_params.shape[1] == 0:
            raise ValueError("Cannot compute PCA with 0-dimensional parameters.")

        # Center the data
        mean = torch.mean(flat_params, dim=0)
        centered_params = flat_params - mean

        # Perform SVD
        try:
            # U: N x N, S: min(N,D), V: D x D (Vh is V.T)
            # We want the principal components, which are the columns of V (or rows of Vh)
            _, _, Vh = torch.linalg.svd(centered_params, full_matrices=False) # pylint:disable=not-callable
        except torch._C._LinAlgError as e: #type:ignore
            print(f"Warning: SVD failed for PCA ({e}). Returning random orthogonal directions.")
            dim = flat_params.shape[1]
            d1 = torch.randn(dim, device=self.device)
            d1 = torch.nn.functional.normalize(d1, dim=0)
            if n_components == 1: return [d1]
            d2 = torch.randn(dim, device=self.device)
            d2 -= torch.dot(d2, d1) * d1
            d2 = torch.nn.functional.normalize(d2, dim=0)
            return [d1, d2]


        # Vh rows are the principal components (eigenvectors of covariance matrix)
        directions = [Vh[i, :].clone() for i in range(n_components)]
        #print("PCA directions computed.")
        return directions

    @torch.no_grad()
    def _evaluate_loss_at_params(self, params_list, closure_eval, current_model_params):
        """Temporarily sets model parameters, evaluates closure, and restores original params."""
        original_params_copy = [p.clone() for p in current_model_params]
        _set_params(current_model_params, params_list)
        loss = closure_eval() # Closure should handle its own zero_grad and forward pass
        _set_params(current_model_params, original_params_copy) # Restore original params
        return loss.item()

    def _perform_visualization(self, closure_eval):
        """Performs subspace finding, grid search, ARS, and visualization."""
        #print("\n--- Starting Visualization Phase ---")
        if len(self.param_history) < 2:
            print("Warning: Not enough history points for visualization. Skipping.")
            # Clear history and continue optimization from the last point
            self.param_history.clear()
            self.grad_history.clear()
            self.loss_history.clear()
            return self.param_groups[0]['params'] # Return current params

        # Get current model parameters (the ones we need to evaluate loss with)
        current_model_params = list(self.param_groups[0]['params'])
        center_params_flat = _flatten_params(current_model_params).to(self.device)

        # 1. Find Subspace Directions
        if self.subspace_method == 'active_subspace':
            try:
                directions_flat = self._get_active_subspace_directions(n_components=2)
            except Exception as e:
                print(f"Error in Active Subspace ({e}), falling back to PCA on parameters.")
                directions_flat = self._get_pca_directions(n_components=2)
        elif self.subspace_method == 'pca_params':
            directions_flat = self._get_pca_directions(n_components=2)
        else: # Should not happen due to init check
            raise ValueError(f"Unknown subspace method: {self.subspace_method}")

        d1_flat, d2_flat = directions_flat[0], directions_flat[1]

        # Project trajectory onto the 2D plane
        trajectory_alphas = []
        trajectory_betas = []
        param_history_flat = [_flatten_params(p_list).to(self.device) for p_list in self.param_history]

        for p_flat in param_history_flat:

            delta = p_flat - center_params_flat

            trajectory_alphas.append(torch.dot(delta, d1_flat).item())
            trajectory_betas.append(torch.dot(delta, d2_flat).item())

        # 2. Determine Initial Grid Range
        min_alpha, max_alpha = min(trajectory_alphas), max(trajectory_alphas)
        min_beta, max_beta = min(trajectory_betas), max(trajectory_betas)

        # Add buffer if trajectory is collapsed (e.g., single point projected)
        if abs(max_alpha - min_alpha) < 1e-6:
            max_alpha += 0.5
            min_alpha -= 0.5
        if abs(max_beta - min_beta) < 1e-6:
            max_beta += 0.5
            min_beta -= 0.5

        range_alpha = (max_alpha - min_alpha) * self.range_extension_factor
        range_beta = (max_beta - min_beta) * self.range_extension_factor
        center_alpha = (max_alpha + min_alpha) / 2.0
        center_beta = (max_beta + min_beta) / 2.0

        current_alpha_min = center_alpha - range_alpha / 2.0
        current_alpha_max = center_alpha + range_alpha / 2.0
        current_beta_min = center_beta - range_beta / 2.0
        current_beta_max = center_beta + range_beta / 2.0

        best_params_viz_flat = center_params_flat # Start assuming center is best
        # Use loss at center (last point of optimization) as initial best loss
        best_loss_viz = self.loss_history[-1] if self.loss_history else float('inf')


        # Store evaluated points for contour plot
        eval_alphas = []
        eval_betas = []
        eval_losses = []

        # Unflatten directions once for efficiency
        d1_unflat = _unflatten_params(d1_flat, current_model_params)
        d2_unflat = _unflatten_params(d2_flat, current_model_params)
        center_params_unflat = _unflatten_params(center_params_flat, current_model_params)


        # --- Grid Search with Range Extension ---
        #print("Performing Grid Search...")
        for extension_iter in range(self.max_range_extensions + 1): # +1 allows initial grid
            alpha_vals = torch.linspace(current_alpha_min, current_alpha_max, self.grid_resolution, device='cpu')
            beta_vals = torch.linspace(current_beta_min, current_beta_max, self.grid_resolution, device='cpu')

            grid_points_to_eval = [] # List of (alpha, beta, params_flat)
            current_grid_alphas = []
            current_grid_betas = []

            # Prepare all parameter sets for the current grid
            for alpha in alpha_vals:
                for beta in beta_vals:
                    # Only evaluate points not already evaluated in previous extensions
                    # (A simple check: if it's inside the *previous* grid boundaries, skip.
                    # More robust would be to check against eval_alphas/betas list)
                    is_new = True
                    if extension_iter > 0:
                        if (prev_alpha_max > alpha > prev_alpha_min and # type:ignore # noqa:F821
                            prev_beta_max > beta > prev_beta_min): # type:ignore # noqa:F821
                            is_new = False

                    if is_new:
                        alpha_dev, beta_dev = alpha.to(self.device), beta.to(self.device)
                        # Calculate params in full space: p = p_center + alpha*d1 + beta*d2
                        # Perform calculation with unflattened tensors for correct structure
                        new_params_unflat = []
                        with torch.no_grad():
                            for p_c, d1_p, d2_p in zip(center_params_unflat, d1_unflat, d2_unflat):
                                new_params_unflat.append(p_c + alpha_dev * d1_p + beta_dev * d2_p)

                        grid_points_to_eval.append((alpha.item(), beta.item(), new_params_unflat))
                        current_grid_alphas.append(alpha.item())
                        current_grid_betas.append(beta.item())


            # Evaluate losses for the new grid points (potentially in batches)
            #print(f"Evaluating {len(grid_points_to_eval)} new grid points...")
            new_losses = []
            batch_size = self.viz_batch_size if self.viz_batch_size else len(grid_points_to_eval)
            if batch_size <= 0: batch_size = len(grid_points_to_eval) # Handle edge case

            for i in range(0, len(grid_points_to_eval), batch_size):
                batch = grid_points_to_eval[i : i + batch_size]
                for alpha_item, beta_item, params_to_eval_unflat in batch:
                    loss = self._evaluate_loss_at_params(params_to_eval_unflat, closure_eval, current_model_params)
                    new_losses.append(loss)
                    eval_alphas.append(alpha_item)
                    eval_betas.append(beta_item)
                    eval_losses.append(loss)

                    # Update best point found so far
                    if loss < best_loss_viz:
                        best_loss_viz = loss
                        # Store the *flattened* best parameters corresponding to this loss
                        best_params_viz_flat = _flatten_params(params_to_eval_unflat).to(self.device)
                        best_alpha_viz, best_beta_viz = alpha_item, beta_item
                        #print(f"  New best grid point: Loss={best_loss_viz:.4f} at (a={best_alpha_viz:.3f}, b={best_beta_viz:.3f})")


            # Check if best point is on the edge and if we should extend

            if extension_iter < self.max_range_extensions and 'best_alpha_viz' in locals():
                is_on_edge = (
                    abs(best_alpha_viz - current_alpha_min) < 1e-5 or # type:ignore # pylint:disable=possibly-used-before-assignment
                    abs(best_alpha_viz - current_alpha_max) < 1e-5 or # type:ignore
                    abs(best_beta_viz - current_beta_min) < 1e-5 or # type:ignore # pylint:disable=possibly-used-before-assignment
                    abs(best_beta_viz - current_beta_max) < 1e-5 # type:ignore
                )
                if is_on_edge:

                    print(f"Best point found on edge. Extending grid (Iteration {extension_iter + 1})...")
                    # Store current boundaries to check for already evaluated points next iter
                    prev_alpha_min, prev_alpha_max = current_alpha_min, current_alpha_max
                    prev_beta_min, prev_beta_max = current_beta_min, current_beta_max
                    # Extend bounds
                    range_alpha *= self.range_extension_factor
                    range_beta *= self.range_extension_factor
                    # Re-center on the *current best point* found on the edge
                    center_alpha, center_beta = best_alpha_viz, best_beta_viz # type:ignore
                    current_alpha_min = center_alpha - range_alpha / 2.0
                    current_alpha_max = center_alpha + range_alpha / 2.0
                    current_beta_min = center_beta - range_beta / 2.0
                    current_beta_max = center_beta + range_beta / 2.0
                else:
                     # Best not on edge, stop extending
                    break
            elif extension_iter == self.max_range_extensions:
                print("Reached max grid extensions.")
                break # Stop after max extensions regardless
            else:
                 # Initial grid done, or no best point found (unlikely), stop extending
                break


        # --- Accelerated Random Search (ARS) ---
        #print(f"Performing Accelerated Random Search ({self.ars_samples} samples)...")
        ars_alphas = (torch.rand(self.ars_samples, device='cpu') * (current_alpha_max - current_alpha_min) + current_alpha_min).tolist()
        ars_betas = (torch.rand(self.ars_samples, device='cpu') * (current_beta_max - current_beta_min) + current_beta_min).tolist()

        ars_points_to_eval = []
        for alpha, beta in zip(ars_alphas, ars_betas):
            alpha_dev, beta_dev = torch.tensor(alpha, device=self.device), torch.tensor(beta, device=self.device)
            new_params_unflat = []
            with torch.no_grad():
                for p_c, d1_p, d2_p in zip(center_params_unflat, d1_unflat, d2_unflat):
                    new_params_unflat.append(p_c + alpha_dev * d1_p + beta_dev * d2_p)
            ars_points_to_eval.append((alpha, beta, new_params_unflat))

        # Evaluate ARS points
        batch_size = self.viz_batch_size if self.viz_batch_size else len(ars_points_to_eval)
        if batch_size <= 0: batch_size = len(ars_points_to_eval)

        for i in range(0, len(ars_points_to_eval), batch_size):
            batch = ars_points_to_eval[i : i + batch_size]
            for alpha_item, beta_item, params_to_eval_unflat in batch:
                loss = self._evaluate_loss_at_params(params_to_eval_unflat, closure_eval, current_model_params)
                eval_alphas.append(alpha_item)
                eval_betas.append(beta_item)
                eval_losses.append(loss)

                # Update best point found so far
                if loss < best_loss_viz:
                    best_loss_viz = loss
                    best_params_viz_flat = _flatten_params(params_to_eval_unflat).to(self.device)
                    best_alpha_viz, best_beta_viz = alpha_item, beta_item
                    #print(f"  New best ARS point: Loss={best_loss_viz:.4f} at (a={best_alpha_viz:.3f}, b={best_beta_viz:.3f})")


        # 4. Generate Visualization
        #print("Generating visualization frame...")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use tricontourf for scattered data
        if len(eval_alphas) > 2:
            # Filter out inf/nan losses which break contour plots
            valid_indices = [i for i, l in enumerate(eval_losses) if math.isfinite(l)]
            if len(valid_indices) > 2:
                plot_alphas = np.array(eval_alphas)[valid_indices]
                plot_betas = np.array(eval_betas)[valid_indices]
                plot_losses = np.array(eval_losses)[valid_indices]

                # Determine contour levels (log scale often looks better for loss)
                min_loss_plot = np.min(plot_losses)
                max_loss_plot = np.percentile(plot_losses, 98) # Avoid extreme outliers dominating scale
                levels = np.logspace(np.log10(min_loss_plot + 1e-9), np.log10(max_loss_plot + 1e-9), 20) if min_loss_plot > 0 else np.linspace(min_loss_plot, max_loss_plot, 20)

                contour = ax.tricontourf(plot_alphas, plot_betas, plot_losses, levels=levels, cmap='viridis_r', extend='max') # LogNorm can also be used
                fig.colorbar(contour, ax=ax, label='Loss')
            else:
                print("Warning: Not enough valid finite loss points to create contour plot.")
                ax.scatter(eval_alphas, eval_betas, c=eval_losses, cmap='viridis_r')

        else:
            ax.scatter(eval_alphas, eval_betas, c=eval_losses, cmap='viridis_r')


        # Plot trajectory
        ax.plot(trajectory_alphas, trajectory_betas, 'r-o', markersize=4, linewidth=1.5, label='Optimization Path')
        ax.plot(trajectory_alphas[0], trajectory_betas[0], 'go', markersize=8, label='Start') # Start of opt cycle
        ax.plot(trajectory_alphas[-1], trajectory_betas[-1], 'yo', markersize=8, label='End (Center)') # End of opt cycle (center of viz)

        # Plot best point found in visualization
        if 'best_alpha_viz' in locals():
            ax.plot(best_alpha_viz, best_beta_viz, 'w*', markersize=12, label=f'Best Viz (L={best_loss_viz:.3f})') #type:ignore

        ax.set_xlabel("Direction 1 (alpha)")
        ax.set_ylabel("Direction 2 (beta)")
        ax.set_title(f"Loss Landscape Projection (Cycle {len(self.visualization_frames) + 1})")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # Save plot to buffer -> PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frame = Image.open(buf)
        self.visualization_frames.append(frame.copy()) # Store a copy
        buf.close()
        plt.close(fig) # Close the plot to free memory

        print(f"--- Visualization Phase Complete. Best loss found: {best_loss_viz:.4f} ---")

        # 5. Prepare for Restart
        # Clear history
        self.param_history.clear()
        self.grad_history.clear()
        self.loss_history.clear()

        # Return the best parameters found during visualization for the next cycle
        # Make sure they are unflattened correctly
        best_params_viz_unflat = _unflatten_params(best_params_viz_flat, current_model_params)
        return best_params_viz_unflat


    def step(self, closure): # pylint:disable=signature-differs
        """Performs a single optimization step OR a visualization cycle.

        Args:
            closure (callable): closure with backward argument
        """
        # --- Check if Visualization Phase is Due ---
        if self.opt_step_count >= self.n_steps_opt:
            def zo_closure(): return closure(False)
            # Perform visualization and get the new starting parameters
            best_params_viz = self._perform_visualization(zo_closure)

            # Set the model parameters to the best ones found
            current_model_params = list(self.param_groups[0]['params'])
            _set_params(current_model_params, best_params_viz)

             # Reset the state of the base optimizer (e.g., Adam's momentum buffers)
             # This is important because we jumped to a new location.
            for d in self.base_optimizer.state.values():
                d.clear()

            # Re-initialize state if necessary (some optimizers might require this)
            # For Adam, clearing state might be sufficient, but re-init ensures clean start
            # self.base_optimizer = self.base_optimizer.__class__(
            #     self.param_groups, **self.base_optimizer.defaults
            # )


            # Reset optimization step counter
            self.opt_step_count = 0
            # We don't return loss here as the step was the visualization


        # --- Standard Optimization Step Phase ---
        # Need to calculate loss and gradients *for the current step*
        # The base optimizer requires loss for some algorithms (like LBFGS)
        # and gradients must be computed *before* base_optimizer.step()

        # Closure computes the loss and implicitly calls backward()
        loss = closure()

        if loss is None:
            # Should not happen if closure is correct, but handle defensively
            print("Warning: closure did not return a loss value during optimization step.")
            return None


        # Store history *before* the base optimizer modifies parameters
        # Use deep copies to avoid aliasing
        current_params_list = [p.clone().detach() for p in self.param_groups[0]['params']]
        self.param_history.append(_flatten_params(current_params_list))
        # Gradients should exist now after loss.backward() called within closure
        current_grads_flat = _flatten_grads(self.param_groups[0]['params']).clone().detach()
        self.grad_history.append(current_grads_flat)
        self.loss_history.append(loss.item())

        # Perform the actual optimization step using the base optimizer
        self.base_optimizer.step() # Don't pass closure here, gradients are already computed

        self.opt_step_count += 1

        return loss

# --- Example Usage ---

if __name__ == '__main__':

    def rosenbrock_torch(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)


    target = torch.randn(32,32)
    def inverse_inverse(x: torch.Tensor) -> torch.Tensor:
        inv, _ = torch.linalg.inv_ex(x.view(32,32)) # pylint:disable=not-callable
        return torch.nn.functional.mse_loss(inv, target)

    # Example Model and Data
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Simple quadratic objective: f(x, y) = (x-3)^2 + (y+2)^2
    class SimpleObjective(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.Parameter(torch.randn(1024))

        def forward(self):
            # Target is (3, -2)
            return inverse_inverse(self.params)

    model = SimpleObjective().to(device)
    print(f"Initial parameters: {model.params.data.cpu().numpy()}")

    # Create the VisualizingOptimizer
    optimizer = VisualizingOptimizer(
        model.parameters(),
        base_optimizer_cls=optim.Adam, # Use Adam as the underlying optimizer
        lr=0.001,                      # Learning rate for Adam
        n_steps_opt=100,              # Optimize for 25 steps
        subspace_method='active_subspace', # Use gradient info
        grid_resolution=50,          # 25x25 grid
        ars_samples=1000,             # 200 random samples
        range_extension_factor=1.8,  # Extend range significantly if needed
        max_range_extensions=10,      # Allow up to 2 extensions
        device=device
    )

    num_cycles = 10
    total_steps = num_cycles * optimizer.n_steps_opt # Approximate total optimization steps

    def training_closure(backward=True):
        loss = model()
        if backward:
            optimizer.zero_grad()
            loss.backward()
        return loss


    print("\nStarting optimization cycles...")
    pbar = tqdm(total=total_steps, desc="Optimizing")
    for i in range(total_steps):
        loss = optimizer.step(training_closure)

        if loss is not None: # Only log loss from actual optimization steps
            pbar.set_postfix(loss=f"{loss.item():.4f}", cycle=(i // optimizer.n_steps_opt) + 1)
        pbar.update(1)

        # Check if a visualization just happened (step count reset)
        if optimizer.opt_step_count == 0 and i > 0:
            pbar.write(f"Completed visualization cycle after step {i+1}.")
            pbar.write(f"  Restarting from: {model.params.data.cpu().numpy()}, Loss: {model().item():.4f}")

    pbar.close()
    print("\nOptimization finished.")
    print(f"Final parameters: {model.params.data.cpu().numpy()}")
    print(f"Final loss: {model().item()}")

    # Save frames as GIF
    if optimizer.visualization_frames:
        print(f"\nSaving {len(optimizer.visualization_frames)} visualization frames as 'optimization_viz.gif'...")
        optimizer.visualization_frames[0].save(
            'optimization_viz.gif',
            save_all=True,
            append_images=optimizer.visualization_frames[1:],
            duration=2000, # milliseconds per frame
            loop=0 # loop forever
        )
        print("Done.")
    else:
        print("No visualization frames were generated.")