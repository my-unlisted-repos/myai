"""here is an idea, instead of storing EMAs of outer product diagonal, why not store EMAs of outer product eigenvecs and eigenvals"""
import torch
import torchzero as tz


class EigenGrad(tz.core.TensorListOptimizer):
    def __init__(self, params, lr):
        defaults = dict(lr = lr)
        super().__init__(params, defaults)

        self.eigvec_sum = self.get_params().detach().clone().to_vec().zero_().requires_grad_(False)
        self.eigval_sum = 0

    @torch.no_grad
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        # initialize all buffers
        lr = self.get_group_key('lr')


        params = self.get_params()
        grad = params.grad

        g = grad.to_vec()

        # add eigenvecs and eigenvals
        eigenvalue = torch.dot(g,g)
        eigenvector = g / (torch.sqrt(eigenvalue) + 1e-8)

        self.eigvec_sum += eigenvector
        self.eigval_sum += eigenvalue

        # adagrad
        inv_sqrt_eigvals = 1.0 / ((self.eigval_sum + 1e-8)**0.5)

        precond_grad = (torch.dot(self.eigvec_sum, g) * inv_sqrt_eigvals) * self.eigvec_sum
        params.sub_(grad.from_vec(precond_grad).mul_(lr))

        return loss

class EigenRMSprop(tz.core.TensorListOptimizer):
    def __init__(self, params, lr, betas = (0.99, 0.99)):
        defaults = dict(lr = lr)
        super().__init__(params, defaults)

        self.eigvec_ema = None
        self.eigval_ema = None
        self.betas = betas

    @torch.no_grad
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        # initialize all buffers
        lr = self.get_group_key('lr')


        params = self.get_params()
        grad = params.grad

        g = grad.to_vec()

        # add eigenvecs and eigenvals
        eigenvalue = torch.dot(g,g)
        eigenvector = g / (torch.sqrt(eigenvalue) + 1e-8)

        if self.eigvec_ema is None: self.eigvec_ema = eigenvector
        else: self.eigvec_ema.lerp_(eigenvector, self.betas[0])

        if self.eigval_ema is None: self.eigval_ema = eigenvalue
        else: self.eigval_ema.lerp_(eigenvalue, self.betas[1])

        # rmsporp
        inv_sqrt_eigvals = 1.0 / ((self.eigval_ema + 1e-8)**0.5)

        precond_grad = (torch.dot(self.eigvec_ema, g) * inv_sqrt_eigvals) * self.eigvec_ema
        params.sub_(grad.from_vec(precond_grad).mul_(lr))

        return loss

class EigenAdam(tz.core.TensorListOptimizer):
    """_summary_

    Args:
        params (_type_): _description_
        lr (_type_): _description_
        betas (tuple, optional):
            first beta is momentum, 2nd and 3rd are eigvec and eigval betas,
            and eigval beta is used for bias correction but they should probably be the same. Defaults to (0.9, 0.99, 0.99).
        eps (_type_, optional): there are multiple epsilons but this is the adam one. Defaults to 1e-8.
    """
    def __init__(self, params, lr, betas = (0.9, 0.99, 0.99), eps = 1e-8):
        defaults = dict(lr = lr, beta1 = betas[0])
        super().__init__(params, defaults)

        self.grad_ema = None
        self.eigvec_ema = None
        self.eigval_ema = None
        self.vec_beta, self.val_beta = betas[1:]
        self.current_step = 1
        self.eps = eps

    @torch.no_grad
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        # initialize all buffers
        lr = self.get_group_key('lr')
        beta1 = self.get_first_group_key('beta1')

        params = self.get_params()
        grad = params.grad

        g = grad.to_vec()

        # squared averages
        eigenvalue = torch.dot(g,g)
        eigenvector = g / (torch.sqrt(eigenvalue) + 1e-8)

        if self.grad_ema is None: self.grad_ema = torch.zeros_like(g)
        if self.eigvec_ema is None: self.eigvec_ema = torch.zeros_like(eigenvector)
        if self.eigval_ema is None: self.eigval_ema = torch.zeros_like(eigenvalue)

        self.grad_ema.lerp_(g, 1-beta1)
        self.eigvec_ema.lerp_(eigenvector, weight = 1 - self.vec_beta)
        self.eigval_ema.lerp_(eigenvalue, weight = 1 - self.val_beta)

        bias_correction1 = 1 - beta1**self.current_step
        bias_correction2 = 1 - self.val_beta**self.current_step

        # adam
        inv_sqrt_eigvals = 1.0 / (((self.eigval_ema**0.5) / (bias_correction2**0.5 + self.eps)) + 1e-8)

        update = (torch.dot(self.eigvec_ema, self.grad_ema) * inv_sqrt_eigvals) * self.eigvec_ema
        params.sub_(grad.from_vec(update).mul_(lr / bias_correction1))

        self.current_step += 1

        return loss
