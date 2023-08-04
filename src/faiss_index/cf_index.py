from __future__ import print_function
# import os
import time
import numpy as np
from tools.timer import timer

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    graphical_output = True
except ImportError:
    graphical_output = False

import faiss

#################################################################
# Small I/O functions
#################################################################

def ivecs_read(fname):
    m = -1
    with open(fname) as f:
        for line in f:
            sp = line.rsplit(":", 1)
            m = len(sp[1].split(","))
            break

    n = 0
    ids = []
    data_list = []
    with open(fname) as f:
        for line in f:
            sp = line.rsplit(":", 1)
            ids.append(sp[0])
            datas = map(float, sp[1].strip().split(","))
            data_list.extend(datas)
            n += 1

    a = np.ndarray(shape=(n, m), dtype='int32', buffer=np.array(data_list))
    # n = int(n / 10000) * 10000
    n = 159744
    return a[:n].copy(), ids

def fvecs_read(fname):
    data, ids = ivecs_read(fname)
    return (data.view('float32'), ids)


def plot_OperatingPoints(ops, nq, **kwargs):
    ops = ops.optimal_pts
    n = ops.size() * 2 - 1
    pyplot.plot([ops.at( i      / 2).perf for i in range(n)],
                [ops.at((i + 1) / 2).t / nq * 1000 for i in range(n)],
                **kwargs)


#################################################################
# prepare common data for all indexes
#################################################################

def read_data():
    print("load data")
    xb, ids = fvecs_read("data/part-00000")
    xq = xb[:5]

    print("xb shape: {}".format(xb.shape))

    return xb, ids, xq


def flat_l2_index(xb, xq):
    d = xb.shape[1]
    index = faiss.IndexFlatL2(d)  # build the index
    print(index.is_trained)
    index.add(xb)  # add vectors to the index
    print(index.ntotal)

    k = 4  # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k)  # sanity check
    print(I)
    print(D)
    D, I = index.search(xq, k)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries

def ivf_flat_index(xb, xq):
    nlist = 100
    k = 4
    d = xb.shape[1]
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search

    assert not index.is_trained
    index.train(xb)
    assert index.is_trained

    with timer("add base data"):
        index.add(xb)  # add may be a bit slower as well

    with timer("nprobe=1"):
        D, I = index.search(xq, k)  # actual search
    print(I[-5:])  # neighbors of the 5 last queries
    print(D[-5:])

    index.nprobe = 10  # default nprobe is 1, try a few more
    with timer("nprobe=1"):
        D, I = index.search(xq, k)
    print("nprobe=10:")
    print(I[-5:])  # neighbors of the 5 last queries
    print(D[-5:])


def index_factory(xb, xq):
    # indexes that are useful when there is no limitation on memory usage
    unlimited_mem_keys = [
        # "IMI2x10,Flat", "IMI2x11,Flat",
        "IVF4096,Flat", "IVF16384,Flat",
        "PCA64,IMI2x10,Flat"]

    # memory limited to 16 bytes / vector
    keys_mem_16 = [
        'IMI2x10,PQ16', 'IVF4096,PQ16',
        'IMI2x10,PQ8+8', 'OPQ16_64,IMI2x10,PQ16'
    ]

    # limited to 32 bytes / vector
    keys_mem_32 = [
        'IMI2x10,PQ32', 'IVF4096,PQ32', 'IVF16384,PQ32',
        'IMI2x10,PQ16+16',
        'OPQ32,IVF4096,PQ32', 'IVF4096,PQ16+16', 'OPQ16,IMI2x10,PQ16+16'
    ]

    # indexes that can run on the GPU
    keys_gpu = [
        "PCA64,IVF4096,Flat",
        "PCA64,Flat", "Flat", "IVF4096,Flat", "IVF16384,Flat",
        "IVF4096,PQ32"]

    keys_to_test = unlimited_mem_keys
    use_gpu = False

    # remember results from other index types
    op_per_key = []

    # keep track of optimal operating points seen so far
    op = faiss.OperatingPoints()

    for index_key in keys_to_test:
        print("============ key", index_key)

        d = xb.shape[1]
        # make the index described by the key
        index = faiss.index_factory(d, index_key)

        print("[%.3f s] train & add" % (time.time() - t0))

        index.train(xb)
        index.add(xb)

        k = 10
        D_ref, I_ref = index.search(xq, k)
        print(D_ref)
        print(I_ref)
        break

    print("[%.3f s] final result:" % (time.time() - t0))

    op.display()


t0 = time.time()

xb, ids, xq = read_data()

# print(faiss.MatrixStats(xb).comments)
flat_l2_index(xb, xq)
# ivf_flat_index(xb, xq)
#index_factory(xb, xq)