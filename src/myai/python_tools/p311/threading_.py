def limit_execution_time(sec):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    def _time_limiter(func):
        def wrapper_func(*args, **kwargs):
            pool = ThreadPoolExecutor(1)
            future = pool.submit(func, *args, **kwargs)
            try:
                return future.result(sec)
            except TimeoutError:
                print('timeout!')
        return wrapper_func
    return _time_limiter

def limit_execution_time_raise(sec):
    from concurrent.futures import ThreadPoolExecutor
    def _time_limiter(func):
        def wrapper_func(*args, **kwargs):
            pool = ThreadPoolExecutor(1)
            future = pool.submit(func, *args, **kwargs)
            return future.result(sec)
        return wrapper_func
    return _time_limiter