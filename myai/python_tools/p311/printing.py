import inspect
import logging
import os
import sys
import warnings
from collections.abc import Mapping


def pretty_print_dict(d: Mapping) -> None:
    """Pretty prints a recursive dictionary using json."""
    import json
    print(json.dumps(dict(d), indent=4, sort_keys=False))


def printargs(*args, **kwargs) -> None:
    for a in args: print(a)
    mlen = max([len(str(k)) for k in kwargs]) if len(kwargs) > 0 else 0
    for k,v in kwargs.items(): print(f'{k.ljust(mlen)} = {v}')

def print_callable_defaults(c, end = '\n') -> None:
    signature = inspect.signature(c)
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            print(f"{k} = {v.default}", end=end)
        else:
            print(f"{k} = ", end  = end)


class ShutUp:
    """https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print"""
    def __init__(self, enable=True): self.enable = enable
    def __enter__(self):
        if self.enable:
            logging.captureWarnings(True)
            logging.disable(logging.WARNING)
            warnings.filterwarnings("ignore")
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w') # pylint:disable=W1514

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            logging.captureWarnings(False)
            logging.disable(logging.NOTSET)
            sys.stdout.close()
            sys.stdout = self._original_stdout
            warnings.resetwarnings()


def format_number(n):
    """
    Numbers between 0.1 and 1000 are displayed normally: 0.1 -> 0.1.

    Numbers with single trailing digit below 0.1 are converted to scientific notation: 0.01 -> 1e-2.

    All numbers are rounded to three significant digits: 0.008431 -> 0.00843.
    
    Numbers below 0.001 are always converted to scientific notation: 0.0008431 -> 8.431e-4.

    :param n: _description_
    """
    n = float(n)