# pylint:disable=signature-differs, not-callable

import torch


class LeastSquaresSGD(torch.optim.Optimizer):
    """this uses least squares to extrapolate gradient, which would be the same as momentum, but it also integrates loss information to
    generate better gradients. THIS REQUIRES A LOT OF MEMORY SO USE 5000 PARAMETERS AT MAX (in paramwise 5000 per param)

    Args:
        params (params): params.
        lr (float, optional): learning rate. Defaults to 0.01.
        window_size (int, optional): number of gradients and losses to store and use. Defaults to 10.
        adjust_factor (float, optional):
            weight of predicted gradients, if 1 uses predicted gradients only,
            if 0 uses original gradients only, else uses linear interpolation. Defaults to 1.
        epsilon (_type_, optional): for stability. Defaults to 1e-8.
        pos_strategy (str, optional):
            strategy to make loss differential positive

                "min" - add minimum.

                "clamp" - clamp to 0.

                "old" - use old loop with nan to nam

                Defaults to "min".
        normalize_weights (bool, optional):
            whether to normalize loss differential to be in (0, 1) range. Defaults to False.
        crop_AB (bool, optional):
            crops A and B matrix only works in paramwise. Defaults to False.
        paramwise (bool, optional):
            if true performs regression for each param separately otherwise for all params at once. Defaults to False.
        max_norm_diff (float | None, optional):
            clips predicted gradient norm to be no larger than original gradient norm times this. Defaults to 3.
    """
    def __init__(
        self,
        params,
        lr=0.01,
        window_size=10,
        adjust_factor: float = 1,
        epsilon=1e-8,
        pos_strategy="min",
        normalize_weights=False,
        crop_AB=False,
        paramwise=False,
        max_norm_diff:float | None= 3,
    ):
        defaults = dict(lr=lr, window_size=window_size, epsilon=epsilon,adjust_factor=adjust_factor)
        super().__init__(params, defaults)

        # Initialize gradient history and parameter shapes per parameter
        self.gradient_history = []
        self.params:list[torch.Tensor] = [p for group in self.param_groups for p in group['params']]
        self.param_shapes = [p.shape for p in self.params]
        self.param_sizes = [p.numel() for p in self.params]
        self.total_grad_size = sum(self.param_sizes)
        for p in self.params:
            self.gradient_history.append({'grads': []})

        self.pos_strategy = pos_strategy
        self.normalize_weights = normalize_weights

        self.crop_AB = crop_AB
        self.paramwise = paramwise
        self.max_norm_diff = max_norm_diff

        # Initialize single loss history for the optimizer
        self.loss_history = []

    @torch.no_grad
    def step(self, closure):
        if self.paramwise:
            return  self._paramwise_step(closure)

        return self._global_step(closure)

    @torch.no_grad
    def _paramwise_step(self, closure):
        with torch.enable_grad(): loss = closure()
        self.loss_history.append(loss.item())

        # Collect current gradients, flatten them, and store in history
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Flatten the gradient
            flat_grad = p.grad.detach().flatten()
            # Store the gradient
            self.gradient_history[i]['grads'].append(flat_grad)
            # Limit history to window_size
            if len(self.gradient_history[i]['grads']) > self.defaults['window_size']:
                self.gradient_history[i]['grads'].pop(0)

        # Perform step for each parameter
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            grads = self.gradient_history[i]['grads']
            if len(grads) >= self.defaults['window_size'] and len(self.loss_history) >= self.defaults['window_size']:
                # Predict the next gradient
                predicted_grad_flat = self._paramwise_predict_gradient(grads, self.loss_history)
                # Reshape the predicted gradient back to original shape
                predicted_grad = predicted_grad_flat.view(self.param_shapes[i])

                # Adjust the current gradient with the predicted future gradient
                if self.defaults['adjust_factor'] == 1: adjusted_grad = predicted_grad
                else: adjusted_grad = p.grad.lerp(predicted_grad, self.defaults['adjust_factor'])

                if self.max_norm_diff is not None:
                    cnorm = torch.linalg.norm(p.grad)
                    anorm = torch.linalg.norm(adjusted_grad)
                    if anorm / cnorm > self.max_norm_diff:
                        adjusted_grad /= (anorm / cnorm) * self.max_norm_diff

                # Update the parameter
                p.data -= self.defaults['lr'] * adjusted_grad
            else:
                # Not enough history, perform standard SGD update
                p.data -= self.defaults['lr'] * p.grad

        return loss

    @torch.no_grad
    def _paramwise_predict_gradient(self, grads, loss_history):
        window_size = self.defaults['window_size']
        epsilon = self.defaults['epsilon']
        # Get the past gradients and corresponding loss values
        past_grads = grads[-window_size:]
        past_losses = loss_history[-window_size:]

        # Ensure we have enough data
        if len(past_grads) < window_size or len(past_losses) < window_size:
            return torch.zeros_like(past_grads[0])

        # Create feature matrix A and target matrix B
        d = past_grads[0].size(0)
        A = torch.zeros((window_size - 1, d + 1), dtype=past_grads[0].dtype, device=past_grads[0].device)
        B = torch.zeros((window_size - 1, d), dtype=past_grads[0].dtype, device=past_grads[0].device)
        for t in range(window_size - 1):
            A[t, :d] = past_grads[t]
            A[t, d] = past_losses[t]
            B[t] = past_grads[t + 1]

        if self.pos_strategy == 'old':
            weights = []
            for t in range(window_size - 1):
                weight = - (past_losses[t + 1] - past_losses[t]) + epsilon
                weights.append(weight)
            weights = torch.tensor(weights, dtype=A.dtype, device=A.device)
            weights = torch.sqrt(weights).nan_to_num_(1e-6,1e-6,1e-6)

        else:
        # Compute weights based on loss differences with epsilon to avoid zero weights
            past_losses_tensor = torch.tensor(past_losses, dtype=A.dtype, device=A.device)
            loss_diff = past_losses_tensor[1:] - past_losses_tensor[:-1]
            weights = -loss_diff + epsilon
            if weights.min() < 0:
                if self.pos_strategy == 'clamp': weights = torch.clamp(weights, min=0.0)
                elif self.pos_strategy == 'min': weights = weights - weights.min()
            if self.normalize_weights:
                weights -= weights.min()
                weights /= weights.max()

            weights = torch.sqrt(weights)

        # Scale A and B by sqrt(weights)
        if self.crop_AB:
            A_weighted = A[:window_size-1] * weights.unsqueeze(1)
            B_weighted = B[:window_size-1] * weights.unsqueeze(1)
        else:
            A_weighted = A * weights.unsqueeze(1)
            B_weighted = B * weights.unsqueeze(1)

        # Check rank of A_weighted
        rank = torch.linalg.matrix_rank(A_weighted)
        if rank < A_weighted.shape[1]:
            # Add regularization to improve rank
            I = torch.eye(A_weighted.shape[1], dtype=A_weighted.dtype, device=A_weighted.device)
            A_weighted = torch.cat([A_weighted, epsilon * I], dim=0)
            # Create zeros tensor with matching columns to B_weighted
            zeros_B = torch.zeros((I.shape[0], B_weighted.shape[1]), dtype=B_weighted.dtype, device=B_weighted.device)
            B_weighted = torch.cat([B_weighted, zeros_B], dim=0)

        # Solve weighted least squares: A_weighted @ X = B_weighted
        X = torch.linalg.lstsq(A_weighted, B_weighted).solution

        # Predict the next gradient
        current_grad = past_grads[-1]
        current_loss = past_losses[-1]
        A_next = torch.cat([current_grad, torch.tensor([current_loss], dtype=current_grad.dtype, device=current_grad.device)], dim=0).unsqueeze(0)
        predicted_grad = A_next @ X

        return predicted_grad.squeeze()

    @torch.no_grad
    def _global_step(self, closure):
        with torch.enable_grad(): loss = closure()
        self.loss_history.append(loss.item())

        # Collect current gradients, flatten and concatenate them
        current_grads = []
        for p in self.params:
            if p.grad is None:
                continue
            flat_grad = p.grad.detach().flatten()
            current_grads.append(flat_grad)
        if current_grads:
            concat_grad = torch.cat(current_grads)
            # Store the concatenated gradient in history
            self.gradient_history.append(concat_grad)
            # Limit history to window_size
            if len(self.gradient_history) > self.defaults['window_size']:
                self.gradient_history.pop(0)

        # Perform step
        if len(self.gradient_history) >= self.defaults['window_size'] and len(self.loss_history) >= self.defaults['window_size']:
            # Predict the next gradient vector
            predicted_grad_vector = self._global_predict_gradient()
            # Split the predicted gradient vector back to individual parameters
            # predicted_grads = torch.split(predicted_grad_vector, self.param_sizes)
            # Collect current gradients for interpolation
            current_grad_vector = torch.cat(current_grads)
            adjusted_grad_vector = current_grad_vector.lerp(predicted_grad_vector, self.defaults['adjust_factor'])

            if self.max_norm_diff is not None:
                cnorm = torch.linalg.norm(current_grad_vector)
                anorm = torch.linalg.norm(adjusted_grad_vector)
                if anorm / cnorm > self.max_norm_diff:
                    adjusted_grad_vector /= (anorm / cnorm) * self.max_norm_diff

            adjusted_grads = torch.split(adjusted_grad_vector, self.param_sizes)
            # Update each parameter
            for p, adj_grad in zip(self.params, adjusted_grads):
                p.data -= self.defaults['lr'] * adj_grad.view(p.shape)
        else:
            # Not enough history, perform standard SGD update
            for p in self.params:
                if p.grad is not None:
                    p.data -= self.defaults['lr'] * p.grad

    @torch.no_grad
    def _global_predict_gradient(self):
        window_size = self.defaults['window_size']
        epsilon = self.defaults['epsilon']

        # Ensure we have enough history
        if len(self.gradient_history) < window_size or len(self.loss_history) < window_size:
            return torch.zeros(self.total_grad_size, dtype=self.gradient_history[0].dtype, device=self.gradient_history[0].device)

        # Get the past concatenated gradients and corresponding loss values
        past_grads = self.gradient_history[-window_size:]
        past_losses = self.loss_history[-window_size:]

        # Create feature matrix A and target matrix B
        n = window_size - 1
        d = self.total_grad_size
        A = torch.zeros((n, d + 1), dtype=past_grads[0].dtype, device=past_grads[0].device)
        B = torch.zeros((n, d), dtype=past_grads[0].dtype, device=past_grads[0].device)
        for t in range(n):
            A[t, :d] = past_grads[t]
            A[t, d] = past_losses[t]
            B[t] = past_grads[t + 1]

        # Compute weights based on loss differences with epsilon to avoid zero weights
        # loss_tensor = torch.tensor(past_losses, dtype=A.dtype, device=A.device)
        # loss_diff = loss_tensor[1:] - loss_tensor[:-1]
        # weights = -loss_diff + epsilon
        # weights = torch.clamp(weights, min=0.0)
        # weights = torch.sqrt(weights)
        if self.pos_strategy == 'old':
            weights = []
            for t in range(window_size - 1):
                weight = - (past_losses[t + 1] - past_losses[t]) + epsilon
                weights.append(weight)
            weights = torch.tensor(weights, dtype=A.dtype, device=A.device)
            if self.normalize_weights:
                weights -= weights.min()
                weights /= weights.max()
            weights = torch.sqrt(weights).nan_to_num_(1e-6,1e-6,1e-6)

        else:
        # Compute weights based on loss differences with epsilon to avoid zero weights
            past_losses_tensor = torch.tensor(past_losses, dtype=A.dtype, device=A.device)
            loss_diff = past_losses_tensor[1:] - past_losses_tensor[:-1]
            weights = -loss_diff + epsilon
            if weights.min() < 0:
                if self.pos_strategy == 'clamp': weights = torch.clamp(weights, min=0.0)
                elif self.pos_strategy == 'min': weights = weights - weights.min()
            if self.normalize_weights:
                weights -= weights.min()
                weights /= weights.max()
            weights = torch.sqrt(weights)

        loss_tensor = torch.tensor(past_losses, dtype=A.dtype, device=A.device)
        loss_diff = loss_tensor[1:] - loss_tensor[:-1]
        weights = -loss_diff + epsilon
        weights = torch.clamp(weights, min=0.0)
        weights = torch.sqrt(weights)

        # Scale A and B by sqrt(weights)
        A_weighted = A * weights.unsqueeze(1)
        B_weighted = B * weights.unsqueeze(1)

        # Check rank of A_weighted
        rank = torch.linalg.matrix_rank(A_weighted)
        if rank < A_weighted.shape[1]:
            # Add regularization to improve rank
            I = torch.eye(A_weighted.shape[1], dtype=A_weighted.dtype, device=A_weighted.device)
            A_weighted = torch.cat([A_weighted, epsilon * I], dim=0)
            # Create zeros tensor with matching columns to B_weighted
            zeros_B = torch.zeros((I.shape[0], B_weighted.shape[1]), dtype=B_weighted.dtype, device=B_weighted.device)
            B_weighted = torch.cat([B_weighted, zeros_B], dim=0)

        # Solve weighted least squares: A_weighted @ X = B_weighted
        X = torch.linalg.lstsq(A_weighted, B_weighted).solution

        # Predict the next gradient
        current_grad = past_grads[-1]
        current_loss = past_losses[-1]
        A_next = torch.cat([current_grad, torch.tensor([current_loss], dtype=current_grad.dtype, device=current_grad.device)], dim=0).unsqueeze(0)
        predicted_grad = A_next @ X
        return predicted_grad.squeeze()