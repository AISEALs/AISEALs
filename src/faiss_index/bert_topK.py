import time
import numpy as np
from tools.timer import timer
import tqdm
import csv

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
# norm='z_score'
norm = 'norm2'

def ivecs_read(fname):
    m = -1
    total = 0

    with open(fname) as f:
        for line in f:
            sp = line.strip().split(':')
            if total == 1:
                m = len(sp[1].split(","))
            total += 1

    ids = []
    data_list = []
    with open(fname) as f:
        for line in tqdm.tqdm(f, total=total):
            try:
                sp = line.split(":", 1)
                datas = np.array(list(map(float, sp[1].strip().split(","))))
                # if norm == 'z_score':
                #     # if datas.std() < 1e-6:
                #     #     continue
                #     datas = list(map(lambda x: (x-datas.mean())/datas.std(), datas))
                # else:
                #     y = sum(map(lambda x: x*x, datas))
                #     datas = list(map(lambda x: np.float32(x/y), datas))

                data_list.extend(datas)
                ids.append(sp[0])
                # if len(ids) > 1000:
                #     break
            except Exception:
                import traceback
                traceback.print_exc()

    n = int(len(data_list)/m)
    a = np.ndarray(shape=(n, m), dtype='float32', buffer=np.array(data_list))
    # a = np.ndarray(shape=(len(data_list), m), dtype='float')
    ids = list(map(int, ids))
    return a.copy(), np.array(ids)


def fvecs_read(fname):
    data, ids = ivecs_read(fname)
    return data.view('float32'), ids.view('int64')


def plot_OperatingPoints(ops, nq, **kwargs):
    ops = ops.optimal_pts
    n = ops.size() * 2 - 1
    pyplot.plot([ops.at(i / 2).perf for i in range(n)],
                [ops.at((i + 1) / 2).t / nq * 1000 for i in range(n)],
                **kwargs)


#################################################################
# prepare common data for all indexes
#################################################################

def read_data():
    print("load data")
    # file_name = "/Users/jiananliu/work/AISEALs/src/embeddings/bert/output_embedding.txt"
    file_name = "/Users/jiananliu/work/AISEALs/src/embeddings/xxx2vec/data/output_embedding_doc2vec.txt"
    xb, ids = fvecs_read(file_name)
    indexes = np.random.choice(xb.shape[0], 10)
    xq = xb[indexes]
    xq_ids = np.array(ids)[indexes]

    print("xb shape: {}".format(xb.shape))

    return xb, ids, xq, xq_ids


def flat_l2_index(xb, xq, ids, xq_ids):
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)  # build the index
    index2 = faiss.IndexIDMap(index)
    print(index2.is_trained)
    # index.add(xb)  # add vectors to the index
    index2.add_with_ids(xb, ids)
    print(index.ntotal)

    topK = 1000
    indexes = np.random.choice(xb.shape[0], topK)

    k = 20  # we want to see 4 nearest neighbors
    # D, I = index.search(xb[:5], k)  # sanity check
    # print(I)
    # print(D)
    D, I = index2.search(xb[indexes, :], k)  # actual search
    assert ids[33069] == "1172105699587149824"
    print(I[:5])  # neighbors of the 5 first queries
    print(D[:5])
    with open('data/output_doc2vec.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', '文章', 'score', '相似文章'])
        for k, vecs, scores in zip(ids[indexes], I, D):
            url = lambda x: f"https://app.58.com/api/community/detail/entry?aid={x}"
            vec_str = ','.join(list(map(str, vecs)))
            # f.write(str(k) + ':' + vec + '\n')
            print(str(k) + ':' + vec_str + '\n')
            writer.writerows([[url(k), score, url(vec)] for vec, score in zip(vecs, scores)])


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

    keys_to_test = keys_mem_16
    use_gpu = False

    # remember results from other index types
    op_per_key = []

    # keep track of optimal operating points seen so far
    op = faiss.OperatingPoints()

    for index_key in keys_to_test:
        index_key = "IVF1024,Flat"
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

xb, ids, xq, xq_ids = read_data()

print(faiss.MatrixStats(xb).comments)
flat_l2_index(xb, xq, ids, xq_ids)
# ivf_flat_index(xb, xq)
# index_factory(xb, xq)
