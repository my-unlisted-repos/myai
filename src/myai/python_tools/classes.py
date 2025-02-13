import typing as T


def subclasses_recursive(cls:type | T.Any) -> set[type]:
    """Recursively get a set of all subclasses of a class (can pass a type or an object of a type)."""
    if not isinstance(cls, type): cls = type(cls)
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in subclasses_recursive(c)])