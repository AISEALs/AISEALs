from contextlib import contextmanager
import time


@contextmanager
def timer(name, logger=None):
    start = time.time()
    yield
    end = time.time()
    if logger == None:
        print('{} COST: {:.3f}s'.format(name, (end - start)))
    else:
        logger.info('{} COST: {:.3f}s'.format(name, (end - start)))


if __name__ == '__main__':
    with timer("test"):
        time.sleep(10)