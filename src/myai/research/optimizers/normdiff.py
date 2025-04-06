# pylint: disable = not-callable
import torch

class NormGradDiff(torch.optim.Optimizer):
    """new optimizer i thought of

    Args:
        params (_type_): _description_
        lr (_type_): _description_
        update_momentum (_type_, optional): momentum for update. Defaults to 0..
        grad_momentum (_type_, optional): momentum for gradient. Defaults to 0..
        diff_momentum (float, optional): momentum for normalized gradient differencce. Defaults to 0.9.
        norm_momentum (float, optional): momentum for gradient norm. Defaults to 0.0.
        alpha (_type_, optional): multiplier for adding difference to gradient. Defaults to 1..
        use_update (bool, optional): uses update difference and doesn't work. Defaults to False.
        renormalize (bool, optional): whether to normalize update back to tracked gradient norm. Defaults to True.
        rms_beta (float, optional): empirical fisher preconditioner (RMSprop), doesn't work well. Defaults to 0.0.
    """
    def __init__(
        self,
        params,
        lr,
        update_momentum=0.0,
        grad_momentum=0.0,
        diff_momentum=0.9,
        norm_momentum=0.0,
        alpha=1.0,
        use_update=False,
        renormalize=True,
        rms_beta=0.0,
    ):
        defaults = dict(lr = lr, alpha = alpha, update_momentum=update_momentum, grad_momentum=grad_momentum, diff_momentum=diff_momentum, norm_momentum=norm_momentum, use_update=use_update, renormalize=renormalize, rms_beta=rms_beta)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None: continue
                state = self.state[param]

                grad = param.grad

                # gradient momentum
                if group['grad_momentum'] != 0:
                    if 'grad_ema' not in state:
                        state['grad_ema'] = grad.clone()
                    else:
                        state['grad_ema'].lerp_(grad, 1 - group['grad_momentum'])
                        grad = state['grad_ema']

                norm = torch.linalg.vector_norm(grad) + 1e-8

                # initial GD step
                if 'prev_norm' not in state:
                    state['prev_norm'] = norm
                    state['prev_grad'] = grad.clone()

                    if group['rms_beta'] != 0:
                        if 'sq_ema' not in state:
                            state['sq_ema'] = param.grad.pow(2)
                        else:
                            state['sq_ema'].addcmul_(param.grad, param.grad, value = 1 - group['rms_beta'])
                        param.addcdiv_(grad, torch.sqrt(state['sq_ema']) + 1e-6, value = -group['lr'])

                    else:
                        param.sub_(grad, alpha = group['lr'])
                    continue

                # normalize gradients to previous gradients norm
                normalized_grad = grad / (norm / state['prev_norm'])

                # calculate difference between gradients and previous gradients
                # and normalize to gradient norm
                diff = normalized_grad - state['prev_grad']
                diff_norm = torch.linalg.vector_norm(diff)
                diff.div_(diff_norm / state['prev_norm'])

                # difference momentum
                if group['diff_momentum'] != 0:
                    if 'diff_ema' not in state:
                        state['diff_ema'] = diff
                    else:
                        state['diff_ema'].lerp_(diff, 1 - group['diff_momentum'])
                        diff = state['diff_ema']

                # update is grad + difference
                update = normalized_grad + diff.mul_(group['alpha'])

                # normalize update back to actual gradient norm
                if group['renormalize']:
                    update_norm = torch.linalg.vector_norm(update)
                    update.div_(update_norm / norm)

                # update momentum
                if group['update_momentum'] != 0:
                    if 'update_ema' not in state:
                        state['update_ema'] = update
                    else:
                        state['update_ema'].lerp_(update, 1 - group['update_momentum'])
                        update = state['update_ema']

                # empirical fisher diagonal preconditioning (doesn't work well)
                if group['rms_beta'] != 0:
                    if 'sq_ema' not in state:
                        state['sq_ema'] = param.grad.pow(2)
                    else:
                        state['sq_ema'].addcmul_(param.grad, param.grad, value = 1 - group['rms_beta'])
                    update /= torch.sqrt(state['sq_ema']) + 1e-6

                # apply update
                param.sub_(update, alpha = group['lr'])

                # store gradient for next step
                if group['use_update']:
                    state['prev_grad'] = update
                else:
                    state['prev_grad'] = grad

                # store or lerp gradient norm for next step
                if group['norm_momentum'] != 0:
                    if 'prev_norm' not in state:
                        state['prev_norm'] = norm
                    else:
                        state['prev_norm'].lerp_(norm, 1 - group['norm_momentum'])
                else:
                    state['prev_norm'] = norm
        return loss
