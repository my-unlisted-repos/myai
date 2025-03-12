# pylint:disable = signature-differs
import torch, torchzero as tz

class DiagLinear(tz.core.TensorListOptimizer):
    """this minimizes quadratic functions with diagonal hessian in 2 gradient evaluations."""
    def __init__(self, params, lr = 1e-3, y = 'grad', allow_negative_curvature = True):
        defaults = dict(lr = lr)
        super().__init__(params, defaults)
        self.allow_negative_curvature = allow_negative_curvature
        self.y = y

    @torch.no_grad
    def step(self, closure):
        lr = self.get_first_group_key('lr')
        params = self.get_params()

        # evaluate initial gradient
        with torch.enable_grad(): l1 = closure()
        g1 = params.ensure_grad_().grad.clone()

        # move by lr and evaluate new gradient
        p1 = params.clone()
        params.sub_(g1, alpha = lr)
        with torch.enable_grad(): l2 = closure()
        g2 = params.ensure_grad_().grad
        p2 = params

        # difference normalized by difference between parameters
        if self.y == 'grad': y_diff = g1 - g2
        elif self.y == 'loss': y_diff = l1 - l2
        elif self.y == 'sum': y_diff = (g1 + l1) - (g2 + l2)
        elif self.y == 'prod': y_diff = (g1 * l1) - (g2 * l2)
        else: raise ValueError(self.y)
        diff = y_diff / (p1 - p2)

        # solve
        step = g1 / diff

        # set negative curvature to lr
        if not self.allow_negative_curvature: step.masked_fill_(y_diff.sign() != g1.sign(), lr)

        params.set_(p1 - step)
        return l1


class DiagQuadratic(tz.core.TensorListOptimizer):
    """3 closure evaluations per step, evaluates 3 parameters/gradients and fits and minimizes quadratic,
    should converge quickly only on convex funcs with diagonal hessian otherwise it isn't guaranteed to work!
    Also I don't know what this method is actually called. This is like brent method for multiple dimensions."""
    def __init__(
        self,
        params,
        lr = 1e-3,
        y = 'grad',
        allow_negative_curvature = True,
        min_a = 1e-8,

    ):
        defaults = dict(lr = lr)
        self.min_a = min_a
        super().__init__(params, defaults)
        self.allow_negative_curvature = allow_negative_curvature
        self.y = y

    @torch.no_grad
    def step(self, closure):
        lr = self.get_first_group_key('lr')
        params = self.get_params()
        p1 = params.clone()

        # evaluate initial gradient
        with torch.enable_grad(): l1 = closure()
        g1 = params.ensure_grad_().grad.clone()

        # g2
        params.sub_(g1, alpha = lr)
        with torch.enable_grad(): l2 = closure()
        g2 = params.ensure_grad_().grad
        p2 = params.clone()

        # g3
        params.sub_(g2, alpha = lr)
        with torch.enable_grad(): l3 = closure()
        g3 = params.ensure_grad_().grad
        p3 = params.clone()

        if self.y == 'loss':
            g1 = l1; g2 = l2; g3 = l3
        elif self.y == 'sum':
            g1 = g1+l1; g2=g2+l2; g3=g3+l3
        elif self.y == 'prod':
            g1 = g1*l1; g2=g2*l2; g3=g3*l3
        else:
            raise ValueError(self.y)

        # fit quadratic
        a = (p1*(g3-g2) + p2*(g1-g3) + p3*(g2-g1)) / ((p1-p2) * (p1 - p3) * (p2 - p3))
        b = (g2-g1) / (p2-p1) - a*(p1+p2)

        # solve
        quadratic_minimizer = -b / (2 * a)

        # solve linear where a <= min_a
        grad_diff = g1 - g2
        diff = grad_diff.div(p1 - p2)
        linear_step = g1 / diff
        if not self.allow_negative_curvature: linear_step.masked_fill_(grad_diff.sign() != g1.sign(), lr)

        a.nan_to_num_(0,0,0)
        quadratic_minimizer.masked_set_(a <= self.min_a, p1 - linear_step)

        params.set_(quadratic_minimizer)
        return l1
