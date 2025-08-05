import math

import torch
from torch.optim import Optimizer


class PadicAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, p_base=2, quantization_levels=5, adaptive_p=False, p_increase_factor=1.1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not isinstance(p_base, int) or p_base < 2:
            raise ValueError("p_base must be an integer >= 2")
        if not isinstance(quantization_levels, int) or quantization_levels < 1:
            raise ValueError("quantization_levels must be an integer >= 1")
        if not 1.0 <= p_increase_factor:
            raise ValueError("p_increase_factor must be >= 1.0")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        p_base=p_base, quantization_levels=quantization_levels,
                        adaptive_p=adaptive_p, p_increase_factor=p_increase_factor)
        super(PadicAdam, self).__init__(params, defaults)

        self.p_current = p_base # Initialize current p-value
        self.loss_history = []
        self.p_adapt_counter = 0
        self.p_adapt_frequency = 100 # Check loss change every this many steps

    def __setstate__(self, state):
        super(PadicAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def _quantize_gradient(self, gradients, p_val, levels):
        """Quantizes gradients based on p-adic inspired levels."""
        quantized_gradients = []
        for g in gradients:
            if g is None: # Handle sparse gradients if needed
                quantized_gradients.append(None)
                continue

            abs_g = torch.abs(g)
            sorted_indices = torch.argsort(abs_g.flatten(), descending=True)
            flattened_g = g.flatten()
            quantized_g_flat = torch.zeros_like(flattened_g)

            num_elements = flattened_g.numel()
            level_size = num_elements // levels # Roughly equal sized levels

            for level in range(levels):
                start_index = level * level_size
                end_index = (level + 1) * level_size if level < levels - 1 else num_elements
                indices_in_level = sorted_indices[start_index:end_index]

                if len(indices_in_level) > 0:
                    # Simple magnitude-based quantization: level value decreases with level number
                    level_value = (p_val ** (levels - 1 - level)) # Example: decreasing powers of p
                    # Or try: level_value = 1.0 / (p_val ** level) # Decreasing fractions

                    # More robust quantization: calculate average magnitude in the level
                    avg_magnitude = abs_g.flatten()[indices_in_level].mean()
                    level_value = avg_magnitude / (p_val ** level) # Scale by p-adic levels
                    level_value = torch.clamp(level_value, min=1e-6) # Prevent division by zero

                    # Assign quantized value based on original sign
                    quantized_g_flat[indices_in_level] = torch.sign(flattened_g[indices_in_level]) * level_value

            quantized_gradients.append(quantized_g_flat.reshape_as(g))
        return quantized_gradients


    def _adjust_p_adaptive(self, current_loss):
        """Adaptively adjust p_current based on loss change."""
        if not self.loss_history:
            self.loss_history.append(current_loss)
            return

        previous_loss = self.loss_history[-1]
        loss_change = current_loss - previous_loss
        self.loss_history.append(current_loss)

        if self.p_adapt_counter % self.p_adapt_frequency == 0:
            if loss_change > 0 and len(self.loss_history) > 10: # Loss increased, might need coarser updates
                loss_trend = sum([self.loss_history[-i] - self.loss_history[-i-1] for i in range(1, 10)]) / 9 # Average recent loss change
                if loss_trend > 0.001: # If consistently increasing
                    self.p_current = int(self.p_current * self.defaults['p_increase_factor'])
                    self.p_current = max(self.p_current, self.defaults['p_base']) # Ensure p doesn't go below base
                    print(f"PadicAdam: Increasing p to {self.p_current} due to loss increase.")
                    self.loss_history = self.loss_history[-5:] # Reset history a bit after p adjustment
            elif self.p_current > self.defaults['p_base'] and loss_change < -0.0001 and len(self.loss_history) > 10:
                loss_trend = sum([self.loss_history[-i] - self.loss_history[-i-1] for i in range(1, 10)]) / 9
                if loss_trend < -0.001: # If consistently decreasing fast enough, maybe try finer p (optional)
                    # self.p_current = max(self.defaults['p_base'], int(self.p_current / self.defaults['p_increase_factor'])) # Decrease p (optional)
                    pass # For now, only increase p

        self.p_adapt_counter += 1


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
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            param_names = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = torch.zeros(1, dtype=torch.int, device=p.device) if isinstance(p, torch.Tensor) else 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

            beta1, beta2 = group['betas']
            p_val = self.p_current # Use current p value
            quantization_levels = group['quantization_levels']

            # 1. Quantize Gradients (P-adic inspired)
            quantized_grads = self._quantize_gradient(grads, p_val, quantization_levels)

            # 2. Perform Adam update with quantized gradients
            for i, param in enumerate(params_with_grad):
                grad = quantized_grads[i]
                if grad is None: # Handle sparse gradients if needed
                    continue
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                amsgrad = group['amsgrad']
                state = self.state[param]
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()

                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                # Momentum update with quantized gradient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # RMSprop update (using quantized gradient components)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)


        if group['adaptive_p']:
            if loss is not None:
                self._adjust_p_adaptive(loss.item())

        return loss


class Padic(Optimizer):
    """Implements the Padic optimization algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        p (int, optional): prime number for p-adic valuation (default: 2)
        eps (float, optional): term added to gradients to avoid division by zero
            (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, p=2, eps=1e-8):
        if p <= 1:
            raise ValueError(f"Invalid prime p: {p}. Must be a prime number > 1.")
        defaults = dict(lr=lr, p=p, eps=eps)
        super(Padic, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            p_prime = group['p']
            eps = group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Padic does not support sparse gradients")

                # Compute p-adic valuation and scaling factor
                abs_grad = torch.abs(grad) + eps
                log_p = math.log(p_prime)
                log_p_abs_grad = torch.log(abs_grad) / log_p
                v = torch.floor(log_p_abs_grad)
                scaling_factor = torch.pow(p_prime, -v)  # Element-wise p^{-v}

                # Apply scaled update
                param.data.add_(-lr * grad * scaling_factor)

        return loss