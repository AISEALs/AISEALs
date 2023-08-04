import numpy as np

data = np.array(range(1, 10)).reshape([3 ,3])

rng = np.random.RandomState(0)
row_idx = rng.permutation(3)
col_idx = rng.permutation(3)

data2 = data[row_idx][:, col_idx]