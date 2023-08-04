from contextlib import contextmanager
import time


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST: {}'.format(name, (end - start) * 1000))
