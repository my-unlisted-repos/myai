import torch
from torch.optim import Optimizer

class HookeJeeves(Optimizer):
    def __init__(self, params, lr=1.0, step_reduction_factor=0.5):
        if lr <= 0.0:
            raise ValueError("Step size must be positive.")
        if not 0.0 < step_reduction_factor < 1.0:
            raise ValueError("Step reduction factor must be in (0, 1).")

        defaults = dict(lr=lr, step_reduction_factor=step_reduction_factor)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group['current_step_size'] = lr
            group['base_point'] = self._clone_param_group(group) # Store base point parameters

    def _clone_param_group(self, group):
        """Clones the parameters in a parameter group."""
        cloned_group = []
        for p in group['params']:
            cloned_group.append(p.clone().detach().requires_grad_(False)) # Detach and no grad
        return cloned_group

    def _set_params_to(self, group, params_list):
        """Sets the parameters in a parameter group to the given parameter list."""
        for i, p in enumerate(group['params']):
            p.data.copy_(params_list[i].data)

    def _get_loss(self, closure, params_like):
        """Evaluates the loss with given parameters."""
        with torch.no_grad(): # Crucial: No gradients during direct search
            original_params = self._clone_param_group(self.param_groups[0]) # Assuming single group for now, adapt if needed
            self._set_params_to(self.param_groups[0], params_like)
            loss = closure(False) # Evaluate loss with modified parameters
            self._set_params_to(self.param_groups[0], original_params) # Restore original params
        return loss

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step."""
        if closure is None:
            raise ValueError("Closure is required for StochasticHookeJeeves")

        loss = None
        for group in self.param_groups:
            current_step_size = group['current_step_size']
            step_reduction_factor = group['step_reduction_factor']
            base_point_params = group['base_point']
            trial_point_params = self._clone_param_group(group) # Start trial point at base point

            improved_in_exploration = False

            # Exploratory Moves
            for param_idx, p in enumerate(group['params']):

                param_elements = p.numel()


                for element_idx in range(param_elements):
                    # Unit directions exploration
                    directions = [1.0, -1.0] # Positive and negative directions

                    for direction in directions:
                        probe_point_params = self._clone_param_group(group)
                        probe_point_param_flat = probe_point_params[param_idx].reshape(-1)
                        probe_point_param_flat[element_idx] += current_step_size * direction

                        probe_loss = self._get_loss(closure, probe_point_params)
                        current_loss = self._get_loss(closure, trial_point_params)

                        if probe_loss < current_loss:
                            trial_point_params = probe_point_params # Move to the improved point
                            improved_in_exploration = True

            # Pattern Move
            if improved_in_exploration:
                pattern_point_params = self._clone_param_group(group)
                for i in range(len(group['params'])):
                    pattern_point_params[i].data.copy_(2 * trial_point_params[i].data - base_point_params[i].data)

                pattern_loss = self._get_loss(closure, pattern_point_params)
                trial_loss = self._get_loss(closure, trial_point_params)

                if pattern_loss < trial_loss:
                    base_point_params = pattern_point_params # Pattern move successful
                else:
                    base_point_params = trial_point_params # Exploratory improvement stands, pattern move failed
            else:
                group['current_step_size'] *= step_reduction_factor # Reduce step size if no exploration improvement

            # Update base point for next iteration
            group['base_point'] = base_point_params
            # Set model parameters to the new base point
            self._set_params_to(group, base_point_params)

            loss = self._get_loss(closure, base_point_params) # Final loss at the base point

        return loss