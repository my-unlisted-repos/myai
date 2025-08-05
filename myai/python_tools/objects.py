import typing as T


def get__name__(obj: T.Any) -> str:
    return obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
