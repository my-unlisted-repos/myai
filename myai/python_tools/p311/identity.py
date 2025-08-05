# def identity[T](x:T) -> T: return x # pylint:disable = E0602
# def identity_kwargs[T](x:T, *args, **kwargs) -> T: return x # pylint:disable = E0602

from typing import TypeVar
T = TypeVar('T')
def identity(x:T) -> T: return x # pylint:disable = E0602
def identity_kwargs(x:T, *args, **kwargs) -> T: return x # pylint:disable = E0602
