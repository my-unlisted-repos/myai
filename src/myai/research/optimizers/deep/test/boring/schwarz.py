import torch
from torch.optim import Optimizer

class SchwarzSGD(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8, separate=True):
        params = list(params)
        if separate:
            groups = [[p] for p in params]
        else:
            groups = [[params]]

        if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
            raise ValueError("groups must be a list of lists of parameters")

        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

        self.groups = []
        self.group_state = []
        param_to_groups = {}

        # Initialize groups and their state
        for group_params in groups:
            # Check that all parameters are in the optimizer's params
            for p in group_params:
                if not any(p is pg for pg in self.param_groups[0]['params']):
                    raise ValueError("Parameter in groups is not in the optimizer's parameters")

            # Create a structure for the group including slices mapping
            group_info = {'params': group_params, 'slices': {}}
            current_idx = 0
            slices = []
            for p in group_params:
                numel = p.numel()
                slices.append((current_idx, current_idx + numel))
                group_info['slices'][p] = (current_idx, current_idx + numel)
                current_idx += numel

            # Initialize state for the group
            device = group_params[0].device if len(group_params) > 0 else 'cpu'
            state = {
                'exp_avg_sq': torch.zeros(current_idx, device=device),
                'beta': beta,
                'eps': eps
            }
            self.groups.append(group_info)
            self.group_state.append(state)

            # Update param_to_groups to track which groups a parameter belongs to
            for p in group_params:
                if p not in param_to_groups:
                    param_to_groups[p] = []
                param_to_groups[p].append((len(self.groups)-1, group_info['slices'][p]))

        # Store param_to_groups as an attribute for quick access
        self.param_to_groups = param_to_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Update each group's exp_avg_sq with current gradients
        for group_idx, group_info in enumerate(self.groups):
            group_params = group_info['params']
            state = self.group_state[group_idx]
            beta = state['beta']
            exp_avg_sq = state['exp_avg_sq']

            # Flatten the gradients of the group's parameters
            grad_parts = []
            for p in group_params:
                if p.grad is None:
                    grad_parts.append(torch.zeros_like(p))
                else:
                    grad_parts.append(p.grad.data.clone())
            grad_flattened = torch.cat([g.contiguous().view(-1) for g in grad_parts]).to(exp_avg_sq.device)

            # Update exp_avg_sq
            exp_avg_sq.mul_(beta).addcmul_(grad_flattened, grad_flattened, value=1 - beta)

        # Apply preconditioned update to each parameter
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if p not in self.param_to_groups:
                    continue  # parameter not in any group, no update (shouldn't happen)

                total_preconditioner = torch.zeros_like(p)
                # Iterate over all groups that contain this parameter
                for (group_idx, (start, end)) in self.param_to_groups[p]:
                    state = self.group_state[group_idx]
                    exp_avg_sq_slice = state['exp_avg_sq'][start:end]
                    # Reshape to match parameter dimensions
                    exp_avg_sq_reshaped = exp_avg_sq_slice.view(p.shape)
                    # Compute preconditioner
                    preconditioner = 1.0 / (exp_avg_sq_reshaped.sqrt() + state['eps'])
                    total_preconditioner += preconditioner.to(p.device)

                # Update parameter
                p.data.add_(-lr * total_preconditioner * grad)

        return loss