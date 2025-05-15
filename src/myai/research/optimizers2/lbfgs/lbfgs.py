from collections import deque
import torch
from torchzero.utils.optimizer import Optimizer
from torchzero.utils import TensorList

class LBFGS(Optimizer):
    def __init__(self, params, lr, history_size=10):
        defaults = dict(lr = lr, history_size = history_size)
        super().__init__(params, defaults)

        self.global_state = self.state[self.param_groups[0]['params'][0]]
        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)
        self.global_state['step'] = 0

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        lr = self.group_vals('lr')
        params = self.get_params()
        grad = params.grad
        prev_params, prev_grad = self.state_vals('prev_params', 'prev_grad', inits=['params', 'grad'])

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        # 1st step
        if self.global_state['step'] == 0:
            # dir = params.grad.sign() # may work fine

            dir = grad
            # initial step size guess taken from pytorch L-BFGS
            lr = min(1.0, 1.0 / grad.abs().global_sum()) * lr # pyright: ignore[reportArgumentType]

        else:
            s_k = params - prev_params
            y_k = grad - prev_grad
            sy_k = s_k.dot(y_k)

            # only add pair if curvature is positive
            if sy_k > 0:
                s_history.append(s_k)
                y_history.append(y_k)
                sy_history.append(sy_k)

            # else:
                # print('negative curvature')

            prev_params.copy_(params)
            prev_grad.copy_(grad)

            # lbfgs part
            # 1st loop
            alpha_list = []
            q = grad.clone()
            z = None
            for s_i, y_i, sy_i in zip(reversed(s_history), reversed(y_history), reversed(sy_history)):
                p_i = 1 / sy_i # this is also denoted as ρ (rho)
                alpha = p_i * s_i.dot(q)
                alpha_list.append(alpha)
                q.sub_(y_i, alpha=alpha) # pyright: ignore[reportArgumentType]

            # calculate z
            # s.y/y.y is also this weird y-looking symbol I couldn't find
            # z is it times q
            # actually H0 = (s.y/y.y) * I, and z = H0 @ q

            z = q * (sy_k / (y_k.dot(y_k)))

            assert z is not None

            # 2nd loop
            for s_i, y_i, sy_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
                p_i = 1 / sy_i
                beta_i = p_i * y_i.dot(z)
                z.add_(s_i, alpha = alpha_i - beta_i)

            dir = z

        params.sub_(dir.mul_(lr))
        self.global_state['step'] += 1
        return loss