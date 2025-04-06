import torch
from torch.optim import Optimizer

class EigenNGD(Optimizer):
    """Natural gradient that stores a history of covariance matrix eigenvectors and eigenvalues (its rank 1 matrix so 1 per).

    set update_freq to other than 1 to make it faster. might need some tuning...

    Args:
        params (_type_): _description_
        lr (_type_, optional): _description_. Defaults to 1e-3.
        momentum (float, optional): momentum for gradient before preconditioning. Defaults to 0.9.
        history_size (int, optional): number of past covariance matrix eigenvectors and eigenvalues to store. Defaults to 10.
        update_freq (int, optional): set to 1 otherwise it won't work. Defaults to 1.
        epsilon (_type_, optional): _description_. Defaults to 1e-8.
        momentum_into_precond (bool, optional): set to false otherwise it is unstable. Defaults to False.
        precond_momentum (int, optional):
            momentum for preconditioning matrices U and V. seems to work well either at 0, or at the same value as momentum. Defaults to 0.
        post_momentum (int, optional):
            momentum for preconditioned gradient, stabilizes it quite well. Defaults to 0.
"""

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.0,
        history_size=10,
        update_freq=1,
        epsilon=1e-8,
        momentum_into_precond=False,
        precond_momentum=0.,
        post_momentum = 0.,
        lowrank=False,
        niter=2,
        extra_q=2,
    ):
        defaults = dict(
            lr=lr,
            history_size=history_size,
            momentum=momentum,
            momentum_into_precond=momentum_into_precond,
            update_freq=update_freq,
            epsilon=epsilon,
            precond_momentum=precond_momentum,
            post_momentum=post_momentum,
            lowrank=lowrank,
            niter=niter,
            extra_q=extra_q,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grad_queue'] = []
                state['U'] = None
                state['S'] = None
                state['step'] = 0

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:

            lr = group['lr']
            history_size = group['history_size']
            update_freq = group['update_freq']
            epsilon = group['epsilon']
            precond_momentum = group['precond_momentum']
            post_momentum = group['post_momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                flattened_grad = grad.detach().clone().view(-1)
                state = self.state[p]
                state['step'] += 1

                # momentum
                if group['momentum'] != 0:
                    if 'ema' not in state:
                        state['ema'] = flattened_grad.clone().detach()
                    else:
                        state['ema'].lerp_(flattened_grad, 1 - group['momentum'])

                    if group['momentum_into_precond']: precond_grad = state['ema']
                    else: precond_grad = flattened_grad

                    flattened_grad = state['ema']
                else:
                    precond_grad = flattened_grad

                # add gradient to queue
                state['grad_queue'].append(precond_grad)
                if len(state['grad_queue']) > history_size:
                    state['grad_queue'].pop(0)

                # update preconditioner every n steps
                if state['step'] % update_freq == 0 and len(state['grad_queue']) > 0:
                    V = torch.stack(state['grad_queue'], dim=1) # d,k
                    success=True
                    try:
                        if group['lowrank']:
                            U, S, _ = torch.svd_lowrank(V, q=len(state['grad_queue'])+group['extra_q'], niter=group['niter'])
                        else:
                            U, S, _ = torch.linalg.svd(V, full_matrices=False) # pylint:disable=not-callable
                    except Exception as e:
                        # when those are None, it falls back to adam
                        state['U'] = None; state['S'] = None
                        U, S = None, None #  shut up pylance
                        success=False

                    if success:
                        assert U is not None and S is not None
                        S.add_(epsilon)

                        # U and V momentum
                        if precond_momentum != 0:
                            if state['U'] is None or state['U'].shape != U.shape:
                                state['U'], state['S'] = U, S
                            else:
                                state['U'].lerp_(U, 1-precond_momentum)
                                state["S"].lerp_(S, 1-precond_momentum)

                        else: state['U'], state['S'] = U, S

                # apply preconditioning
                if state['U'] is not None and state['S'] is not None and (len(state['grad_queue']) == group['history_size'] or group['update_freq'] == 1):
                    if 'adam_ema' in state: del state['adam_ema']
                    U, S = state['U'], state['S']
                    Ut_g = torch.mv(U.t(), flattened_grad)
                    scaled_Ut_g = Ut_g / S
                    precond_grad_flat = torch.mv(U, scaled_Ut_g)
                    precond_grad = precond_grad_flat.view_as(grad)

                    if post_momentum != 0:
                        if 'post_ema' not in state:
                            state['post_ema'] = precond_grad
                        else:
                            state['post_ema'].lerp_(precond_grad, 1-post_momentum)
                        precond_grad = state['post_ema']
                    p.add_(precond_grad,alpha=-lr)
                else:
                    # use adam-like ema so that lr is transferrable
                    g = flattened_grad.view_as(p)
                    if 'adam_ema' not in state: state['adam_ema'] = g.pow(2)
                    else: state['adam_ema'].addcmul_(g,g,value=0.01)
                    p.addcdiv_(g, state['adam_ema'].sqrt().add(epsilon), value=-lr)
                #     p.add_(-lr, grad)
        return loss