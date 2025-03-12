import functools

import numpy as np
import torch
import scipy.optimize
import torchzero as tz

class SigmaTransformer:
    """sigma boy"""
    def __init__(self, bounds, alpha):
        self.bounds = np.array(bounds)
        self.n = len(bounds)
        if self.n < 2:
            raise ValueError("Number of variables must be at least 2")
        self.alpha = alpha
        self.a = alpha / (2 * np.pi)
        self.theta = []

        # Compute theta_j for j=2 to n (stored in self.theta)
        a1, b1 = self.bounds[0]
        a2, b2 = self.bounds[1]
        theta_2 = np.sqrt((b1 - a1)**2 + (b2 - a2)**2) / (2 * self.a)
        self.theta.append(theta_2)

        for j in range(2, self.n):
            a_j, b_j = self.bounds[j]
            theta_j = (b_j - a_j) / self.alpha * self.theta[j-2]
            self.theta.append(theta_j)

    def compute_x(self, theta):
        if self.n < 2:
            return np.array([theta])

        x = np.zeros(self.n)
        alphas = []
        for j in range(len(self.theta)):
            theta_j = self.theta[j]
            beta_j = int(theta / theta_j)
            term = beta_j + 0.5 * ((-1)**(beta_j + 1) + 1)
            alpha_j = (-1)**beta_j * (theta - term * theta_j)
            alphas.append(alpha_j)

        # Compute x1 and x2 using alpha_2 (alphas[0])
        alpha_2 = alphas[0]
        x[0] = self.a * alpha_2 * np.cos(alpha_2)
        x[1] = self.a * alpha_2 * np.sin(alpha_2)

        # Compute x3 to xn using alphas[j-1]
        for j in range(2, self.n):
            a_j, b_j = self.bounds[j]
            alpha_j_val = alphas[j-1]
            theta_j_val = self.theta[j-1]
            x[j] = a_j + (alpha_j_val / theta_j_val) * (b_j - a_j)

        return x

def reduce_function(f, bounds, alpha):
    transformer = SigmaTransformer(bounds, alpha)
    def f_star(theta):
        x = transformer.compute_x(theta)
        return f(x)
    return f_star



def brent(fn, interval):
    res = scipy.optimize.minimize_scalar(fn, bracket = (0, interval))
    return res.x

class OneDimensionalOptimizer(tz.core.TensorListOptimizer):
    """I didn't know about this one"""
    def __init__(self, params, alpha = 0.1, interval = 1000, lb = -100000, ub = 100000, solver = brent):
        super().__init__(params, {})

        self.alpha = alpha
        self.interval = interval
        self.lb = lb
        self.ub = ub
        self.solver = solver

        self.reduced_fn = None

    def objective(self, x, params, closure):
        p = params[0]
        vec = torch.as_tensor(x, dtype = p.dtype, device = p.device)
        params.from_vec_(vec)
        loss = closure(False)
        if self._best is None or loss < self._best: self._best = loss
        return loss

    @torch.no_grad
    def step(self, closure):
        params = self.get_params()
        if self.reduced_fn is None:
            objective = functools.partial(self.objective, params = params, closure = closure)
            self.transformer = SigmaTransformer(([self.lb,self.ub], ) * params.total_numel(), self.alpha)
            def f_star(theta):
                x = self.transformer.compute_x(theta)
                return objective(x)
            self.reduced_fn = f_star

        self._best = None
        v = self.solver(self.reduced_fn, self.interval)
        x = self.transformer.compute_x(v)
        p = params[0]
        params.from_vec_(torch.as_tensor(x, dtype = p.dtype, device = p.device))
        return self._best



# https://github.com/vaseline555/Algorithms-for-Optimization-Python/blob/main/Ch%2003.%20Bracketing.ipynb
class Pt:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def _get_sp_intersection(A: Pt, B: Pt, l: float) -> Pt:
    t = ((A.y - B.y) - l * (A.x - B.x)) / (2 * l)
    return Pt(A.x + t, A.y - t * l)

def shubert_piyavskii(f, a, b, l, eps=1e-5, delta=0.01):
    m = (a + b) / 2
    A, M, B = Pt(a, f(a)), Pt(m, f(m)), Pt(b, f(b))
    pts = [A, _get_sp_intersection(A, M, l), M, _get_sp_intersection(M, B, l), B]
    diff = np.inf
    while diff > eps:
        i = np.argmin([P.y for P in pts])
        P = Pt(pts[i].x, f(pts[i].x))
        diff = P.y - pts[i].y

        P_prev = _get_sp_intersection(pts[i - 1], P, l)
        P_next = _get_sp_intersection(P, pts[i + 1], l)

        del pts[i]
        pts.insert(i, P_next)
        pts.insert(i, P)
        pts.insert(i, P_prev)

    intervals = []
    i = 2 * np.argmin([P.y for P in pts[::2]]) - 1
    for j in range(1, len(pts), 2):
        if pts[j].y < pts[i].y:
            dy = pts[i].y - pts[j].y
            x_lo = max(a, pts[j].x - dy / l)
            x_hi = min(b, pts[j].x + dy / l)
            if intervals:
                if intervals[-1][1] + delta >= x_lo:
                    intervals[-1] = (intervals[-1][0], x_hi)
            else:
                intervals.append((x_lo, x_hi))
    return intervals[-1]

def shubert_piyavskii_solver(f, interval):
    z = shubert_piyavskii(f, 0, interval, 10)
    return z[1]