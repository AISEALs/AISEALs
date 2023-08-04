from multiprocessing import Pool

# 参考 1.matrix select top K.   2. pool.map multiple arguments
# 1.https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices
# 2.https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments

def _top_k(args, k):
    """
    Helper function to process a single row of top_k
    """
    data, row = args
    data, row = zip(*sorted(zip(data, row), reverse=True)[:k])
    return data, row

def top_k(m, k):
    """
    Keep only the top k elements of each row in a csr_matrix
    """
    ml = m.tolil()
    # print("row: {}".format(ml.rows))
    # print("data: {}".format(ml.data))
    with Pool() as p:
        from functools import partial
        ms = p.map(partial(_top_k, k=k), zip(ml.data, ml.rows))
    ml.data, ml.rows = zip(*ms)
    return ml.tocsr()


if __name__ == '__main__':
    import numpy as np
    from scipy.sparse import csr_matrix

    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    m = csr_matrix((data, (row, col)), shape=(3, 3))
    print(top_k(m, 5))
