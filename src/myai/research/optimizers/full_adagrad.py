import torch
import torchzero as tz


class FullAdagrad(tz.core.TensorListOptimizer):
    """full covariance matrix version of Adagrad / RMSprop

    Args:
        params (_type_): _description_
        lr (_type_, optional): _description_. Defaults to 1e-2.
        tikhonov (float, optional): added to diagonal of covariance matrix. Defaults to 1e-8.
        eps (_type_, optional): added to eigenvalues. Defaults to 1e-8.
        beta (float | None, optional): if not None, set to like 0.99 to get RMSprop. Defaults to None.
        decay (float, optional): multiplies covariance matrix by 1-decay on each step, 0.5 works really well. Defaults to 0.
    """
    def __init__(self, params, lr=1e-2, tikhonov=1e-8, eps=1e-8, beta: float | None = None, decay: float = 0):

        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.eps = eps
        self.tikhonov=tikhonov

        p = self.get_params()
        n = p.total_numel()
        self.cov = torch.zeros((n, n), device = p[0].device, dtype = p[0].dtype)
        self.beta = beta
        self.decay = decay
        self.first_step = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lr = self.get_group_key('lr')


        params = self.get_params()
        grad = params.grad

        g = grad.to_vec()
        if self.beta is None or self.first_step: self.cov.add_(torch.outer(g, g))
        else: self.cov.mul_(self.beta).add_(torch.outer(g, g), alpha = 1 - self.beta)

        if self.tikhonov != 0:
            cov_reg = self.cov + torch.eye(self.cov.shape[0], device = params[0].device, dtype = params[0].dtype)

        else:
            cov_reg = self.cov

        if self.decay != 0: self.cov *= 1 - self.decay

        # eigen decomposition for inverse square root
        eigvals, eigvecs = torch.linalg.eigh(cov_reg) # pylint:disable=not-callable
        eigvals.add_(self.eps)
        inv_sqrt_eigvals = 1.0 / torch.sqrt(eigvals)
        inv_sqrt_matrix = eigvecs @ torch.diag(inv_sqrt_eigvals) @ eigvecs.T

        precond_grad = inv_sqrt_matrix @ g
        params.sub_(grad.from_vec(precond_grad).mul_(lr))

        self.first_step = False
        return loss
