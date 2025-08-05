# pylint:disable=undefined-variable
import time
from collections.abc import Generator

from .types_ import SupportsIter


class Progress[V]:
    def __init__(self, iterable:SupportsIter[V], sec:float = 1, enable=True,):
        """Basic progress.

        Args:
            iterable (SupportsIter): iterable
            sec (float, optional): print progress every `sec` seconds. Defaults to 1.
            enable (bool, optional): set to False to disable progress. Defaults to True.
        """
        self.iterable: SupportsIter[V] = iterable
        try:
            self.len = len(self.iterable) # type:ignore
        except Exception:
            self.len = None

        self.sec = sec
        self.enable = enable

    def __iter__(self) -> Generator[V]:
        start = time.time()
        last_print_time = 0
        for i, item in enumerate(self.iterable): # type:ignore
            yield item
            if self.enable:
                t = time.time()
                time_passed = t - start

                # on first iteration print i/len
                if i == 0:
                    if self.len is None: print(f"{i}", end = '          \r')
                    else: print(f"{i} / {self.len}", end = '          \r')
                    last_print_time = t
                else:
                    if t - last_print_time > self.sec:
                        last_print_time = t
                        ops_per_sec = i / max(1e-6, time_passed)
                        if self.len is None:
                            print(f"{i} | {t - start:.2f}s", end = '          \r')
                        else:
                            remaining = (self.len - i) / ops_per_sec
                            print(f"{i}/{self.len} | {t - start:.2f}s/{remaining:.2f}s", end = '          \r')

                # print on last iteration
                if self.len is not None and i == self.len - 1:
                    print(f"{i+1}/{self.len} | {t - start:.2f}s", end = '          \r')

        if self.enable: print()