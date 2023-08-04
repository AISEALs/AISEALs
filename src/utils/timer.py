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


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


if __name__ == '__main__':
    with timer("test"):
        time.sleep(10)