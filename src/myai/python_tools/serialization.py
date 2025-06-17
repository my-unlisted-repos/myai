import typing as T
from collections.abc import Callable, Mapping


SERIALIZEABLE_TYPES = int, float, str, bool
Serializeable = int | float | str | bool

def make_dict_serializeable(
    x: Mapping,
    maxstr=None,
    recursive=False,
    allowed_types=SERIALIZEABLE_TYPES,
    type_handlers: dict[type, Callable[[T.Any], T.Any]] | None = None,
    default_handler:Callable[[T.Any], T.Any] = str,
    raw_strings = True,
) -> dict:
    """Coverts all non serializeable objects in a dictionary to strings and truncates all strings to maxstr length.

    :param x: The dictionary to convert.
    :param maxstr: Maximum string length, defaults to 1000
    :param recursive: Whether to convert nested dictionaries, defaults to False
    :param allowed_types: Types that are allowed to be in the dictionary, defaults to SERIALIZEABLE_TYPES
    :param type_handlers: Callables for specific types that covert them to serializeable objects, defaults to None
    :param default_handler: Callable for types that are not in allowed_types or type_handlers, defaults to str, which coverts them to a string.
    :param raw_strings: Whether to convert all strings to raw strings, defaults to True
    :return: the dictionary.
    """
    x = dict(x)
    for k, v in x.copy().items():
        if isinstance(v, dict) and recursive:
            x[k] = make_dict_serializeable(v, maxstr)
        elif isinstance(v, str):
            x[k] = v[:maxstr]
        elif isinstance(v, allowed_types):
            pass
        elif type_handlers is not None and type(v) in type_handlers:
            x[k] = type_handlers[type(v)](v)
        else:
            x[k] = default_handler(x[k])[:maxstr]

    if raw_strings:
        for k, v in x.copy().items():
            if isinstance(v, str):
                x[k] = f'{v[:maxstr]!r}'[1:-1]
    return x


