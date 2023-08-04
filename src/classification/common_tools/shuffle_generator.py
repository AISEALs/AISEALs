import random
from itertools import islice
import numpy as np


def shuffle_by_buffer(generator, buffer_size):
    while True:
        buffer = list(islice(generator, buffer_size))
        if len(buffer) == 0:
            break
        np.random.shuffle(buffer)
        for item in buffer:
            yield item


# 全局打乱顺序
def shuffle_all_in_cache(generator):
    lst = list(generator)
    random.shuffle(lst)
    print("partition num:{}".format(len(lst)))
    for item in lst:
        yield item
