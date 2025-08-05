import contextlib
import typing as T
import time

@contextlib.contextmanager
def performance_context(name: str | T.Any | None = None, ndigits: int | None = None):
    """shows both time and perf_counter seconds"""
    time_start = time.time()
    perf_counter_start = time.perf_counter()
    yield
    time_took = time.time() - time_start
    perf_counter_took = time.perf_counter() - perf_counter_start
    if name is None: name = "Context"
    if ndigits is not None:
        time_took = round(time_took, ndigits)
        perf_counter_took = round(perf_counter_took, ndigits)
    print(f"{name} took {time_took} s. | {perf_counter_took} perf_counter s.")

@contextlib.contextmanager
def time_context(name: str | T.Any | None = None, ndigits: int | None = None):
    time_start = time.time()
    yield
    time_took = time.time() - time_start
    if name is None: name = "Context"
    if ndigits is not None: time_took = round(time_took, ndigits)
    print(f"{name} took {time_took} seconds")

@contextlib.contextmanager
def perf_counter_context(name: str | T.Any | None = None, ndigits: int | None = None):
    time_start = time.perf_counter()
    yield
    time_took = time.perf_counter() - time_start
    if name is None: name = "Context"
    if ndigits is not None: time_took = round(time_took, ndigits)
    print(f"{name} took {time_took} perf_counter seconds")

class PerfCounter:
    def __init__(self):
        self.times= []

    def step(self):
        self.times.append(time.perf_counter())
