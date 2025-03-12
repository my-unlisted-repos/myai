import torch
from torch.nn import functional as F

from myai.nn.layers.algebraic_linear import algebraic_linear, algebraic_matmul, AlgebraicLinear



def _assert_allclose(x, y):
    assert x.shape == y.shape, f'{x.shape = }, {y.shape = }'
    assert torch.allclose(x, y, rtol=1e-5,atol=1e-5), f'{(x - y).abs().mean() = }'

def _assert_matmul(x, y):
    real = x @ y
    sem = algebraic_matmul(x, y)
    _assert_allclose(real, sem)

def test_algebraic_matmul():
    # vv
    x = torch.randn(10); y = torch.randn(10)
    _assert_matmul(x, y)

    # outer
    x = torch.randn(10, 1); y = torch.randn(1, 4)
    _assert_matmul(x, y)

    # vM
    x = torch.randn(10); y = torch.randn(10, 3)
    _assert_matmul(x, y)

    # Mv
    x = torch.randn(10, 3); y = torch.randn(3)
    _assert_matmul(x, y)

    # MM
    x = torch.randn(12, 5); y = torch.randn(5, 8)
    _assert_matmul(x, y)

    # batched + broadcasting
    x = torch.randn(4, 3, 12, 5); y = torch.randn(3, 5, 8)
    _assert_matmul(x, y)

    # broadcasting2
    x = torch.randn(4, 3, 12, 5); y = torch.randn(1, 5, 8)
    _assert_matmul(x, y)

    # Mv broadcasting
    x = torch.randn(4,3,12,5); y = torch.randn(5)
    _assert_matmul(x, y)

    # vM broadcasting
    x = torch.randn(12); y = torch.randn(4,3,12,5)
    _assert_matmul(x, y)


@torch.no_grad
def _assert_matches_with_torch_linear(W,b):
    out_channels, in_channels = W.shape[0], W.shape[1]

    # vec
    x = torch.randn(in_channels)
    _assert_allclose(F.linear(x, W, b), algebraic_linear(x, W, b)) # pylint:disable=not-callable

    # batched1
    x = torch.randn(32, in_channels)
    _assert_allclose(F.linear(x, W, b), algebraic_linear(x, W, b)) # pylint:disable=not-callable

    # batched2
    x = torch.randn(32, 3, in_channels)
    _assert_allclose(F.linear(x, W, b), algebraic_linear(x, W, b)) # pylint:disable=not-callable

def test_algebraic_linear():
    W = torch.randn(12, 1)
    b = torch.randn(12)
    _assert_matches_with_torch_linear(W, b)

    W = torch.randn(1, 12)
    b = torch.randn(1)
    _assert_matches_with_torch_linear(W, b)

    W = torch.randn(64, 32)
    b = torch.randn(64)
    _assert_matches_with_torch_linear(W, b)


@torch.no_grad
def _assert_module_matches_with_torch_linear(W,b):
    in_channels, out_channels = W.shape[1], W.shape[0]
    torch_linear = torch.nn.Linear(in_channels, out_channels)
    torch_linear.weight.set_(W)
    torch_linear.bias.set_(b)

    alg_linear = AlgebraicLinear(in_channels, out_channels)
    alg_linear.weight.set_(W)
    alg_linear.bias.set_(b) # type:ignore

    # vec
    x = torch.randn(in_channels)
    _assert_allclose(torch_linear(x), alg_linear(x))

    # batched1
    x = torch.randn(32, in_channels)
    _assert_allclose(torch_linear(x), alg_linear(x))

    # batched2
    x = torch.randn(32, 3, in_channels)
    _assert_allclose(torch_linear(x), alg_linear(x))

def test_algebraic_linear_module():
    W = torch.randn(12, 1)
    b = torch.randn(12)
    _assert_module_matches_with_torch_linear(W, b)

    W = torch.randn(1, 12)
    b = torch.randn(1)
    _assert_module_matches_with_torch_linear(W, b)

    W = torch.randn(64, 32)
    b = torch.randn(64)
    _assert_module_matches_with_torch_linear(W, b)