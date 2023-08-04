from scipy import stats
import numpy as np

data = np.fromstring('3.53 1 4.93 50 5.53 60 6.21 70 7.37 80 9.98 90 16.56 100', sep=' ').reshape(7, 2)

stats.boxcox(data[0,])