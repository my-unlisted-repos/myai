from collections import deque
from operator import itemgetter
import torch

from torchzero.core import Transform, Chainable, maybe_chain, Module, Vars
from torchzero.utils import TensorList, as_tensorlist, NumberList


def _get_sk_yk_ysk(
    params: TensorList,
    grad: TensorList,
    prev_params_: TensorList,
    prev_grad_: TensorList,
    damping = False,
    init_damping = 0.99,
    eigval_bounds = (0.01, 1.5)
):

    s_k = params - prev_params_
    y_k = grad - prev_grad_
    ys_k = s_k.dot(y_k)

    if damping:
        # adaptive damping Al-Baali, M.: Quasi-Wolfe conditions for quasi-Newton methods for large-scale optimization. In: 40th Workshop on Large Scale Nonlinear Optimization, Erice, Italy, June 22–July 1 (2004)
        sigma_l, sigma_h = eigval_bounds
        u = ys_k / s_k.dot(s_k)
        if u <= sigma_l < 1: tau = min((1-sigma_l)/(1-u), init_damping)
        elif u >= sigma_h > 1: tau = min((sigma_h-1)/(u-1), init_damping)
        else: tau = init_damping
        y_k = tau * y_k + (1-tau) * s_k
        ys_k = s_k.dot(y_k)

    prev_params_.copy_(params)
    prev_grad_.copy_(grad)

    return s_k, y_k, ys_k


def _update_lbfgs_history_(
    params: TensorList,
    grad: TensorList,
    prev_params_: TensorList,
    prev_grad_: TensorList,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    sy_history: deque[torch.Tensor],
    damping = False,
    init_damping = 0.99,
    eigval_bounds = (0.01, 1.5)
):
    s_k, y_k, ys_k = _get_sk_yk_ysk(
        params=params,
        grad=grad,
        prev_params_=prev_params_,
        prev_grad_=prev_grad_,
        damping=damping,
        init_damping=init_damping,
        eigval_bounds=eigval_bounds,
    )
    # only add pair if curvature is positive
    if ys_k > 1e-10:
        s_history.append(s_k)
        y_history.append(y_k)
        sy_history.append(ys_k)

    #else:
        # print(f'negative curvature: {sy_k}')

    return y_k, ys_k

def lbfgs(
    tensors_: TensorList,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    sy_history: deque[torch.Tensor],
    y_k: TensorList,
    ys_k: torch.Tensor,
    step: int,
):
    if step == 0 or len(s_history) == 0:
        # dir = params.grad.sign() # may work fine

        # initial step size guess taken from pytorch L-BFGS
        return tensors_.mul_(min(1.0, 1.0 / tensors_.abs().global_sum())) # pyright: ignore[reportArgumentType]

    else:
        # 1st loop
        alpha_list = []
        q = tensors_.clone()
        z = None
        for s_i, y_i, ys_i in zip(reversed(s_history), reversed(y_history), reversed(sy_history)):
            p_i = 1 / ys_i # this is also denoted as ρ (rho)
            alpha = p_i * s_i.dot(q)
            alpha_list.append(alpha)
            q.sub_(y_i, alpha=alpha) # pyright: ignore[reportArgumentType]

        # calculate z
        # s.y/y.y is also this weird y-looking symbol I couldn't find
        # z is it times q
        # actually H0 = (s.y/y.y) * I, and z = H0 @ q
        z = q * (ys_k / (y_k.dot(y_k)))

        assert z is not None

        # 2nd loop
        for s_i, y_i, ys_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
            p_i = 1 / ys_i
            beta_i = p_i * y_i.dot(z)
            z.add_(s_i, alpha = alpha_i - beta_i)

        return z

def _lerp_params_update_(
    self_: Module,
    params: list[torch.Tensor],
    update: list[torch.Tensor],
    params_beta: list[float | None],
    grads_beta: list[float | None],
):
    for i, (p, u, p_beta, u_beta) in enumerate(zip(params.copy(), update.copy(), params_beta, grads_beta)):
        if p_beta is not None or u_beta is not None:
            state = self_.state[p]

            if p_beta is not None:
                if 'param_ema' not in state: state['param_ema'] = p.clone()
                else: state['param_ema'].lerp_(p, 1-p_beta)
                params[i] = state['param_ema']

            if u_beta is not None:
                if 'grad_ema' not in state: state['grad_ema'] = u.clone()
                else: state['grad_ema'].lerp_(u, 1-u_beta)
                update[i] = state['grad_ema']

    return TensorList(params), TensorList(update)

class LBFGS(Module):
    def __init__(
        self,
        history_size=10,
        tol: float | None = 1e-10,
        damping: bool = False,
        init_damping=0.9,
        eigval_bounds=(0.5, 50),
        params_beta = None,
        grads_beta = None,
        update_freq = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(history_size=history_size, tol=tol, damping=damping, init_damping=init_damping, eigval_bounds=eigval_bounds, params_beta=params_beta, grads_beta=grads_beta, update_freq=update_freq)
        super().__init__(defaults)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)
        self.global_state['step'] = 0

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, vars):
        params = as_tensorlist(vars.params)
        update = as_tensorlist(vars.get_update())
        prev_params, prev_grad = self.get_state('prev_params', 'prev_grad', params=params, cls=TensorList, init=[params, update])

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        tol, damping, init_damping, eigval_bounds, update_freq = itemgetter(
            'tol', 'damping', 'init_damping', 'eigval_bounds', 'update_freq')(self.settings[params[0]])
        params_beta, grads_beta = self.get_settings('params_beta', 'grads_beta', params=params, cls=NumberList)

        l_params, l_update = _lerp_params_update_(self, params, update, params_beta, grads_beta)

        # update effective preconditioning state
        if self.global_state['step'] % update_freq == 0:
            y_k, ys_k = _update_lbfgs_history_(
                params=l_params,
                grad=l_update,
                prev_params_=prev_params,
                prev_grad_=prev_grad,
                s_history=s_history,
                y_history=y_history,
                sy_history=sy_history,
                damping=damping,
                init_damping=init_damping,
                eigval_bounds=eigval_bounds,
            )
        else:
            s_k, y_k, ys_k = _get_sk_yk_ysk(
                params=l_params,
                grad=l_update,
                prev_params_=prev_params,
                prev_grad_=prev_grad,
                damping=damping,
                init_damping=init_damping,
                eigval_bounds=eigval_bounds,
            )

        if tol is not None and self.global_state['step'] != 0: # it will be 0 on 1st step
            if y_k.abs().global_max() <= tol: return vars

        # step with inner module before applying preconditioner
        if self.children:
            inner_module = self.children['inner']
            inner_vars = inner_module.step(vars.clone(clone_update=False))
            vars.update_attrs_from_clone_(inner_vars)
            update = inner_vars.update
            assert update is not None

        # precondition
        dir = lbfgs(
            tensors_=as_tensorlist(update),
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
            y_k=y_k,
            ys_k=ys_k,
            step=self.global_state['step']
        )

        self.global_state['step'] += 1
        vars.update = dir
        return vars

