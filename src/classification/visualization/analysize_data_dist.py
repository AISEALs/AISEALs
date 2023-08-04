import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

dir = '/Users/jiananliu/work/python/AISEALs/data/text_classification/youliao_train/'
files = os.listdir(dir)
tmpdate=[]

for fileName in files:
    examples = list(open(dir+fileName, "r",encoding='UTF-8').readlines())
    examples = [s.strip() for s in examples]
    examples = [len(x.split(" ")) for x in examples]

    # Split by words
    tx_text =  examples
    tmpdate.extend(tx_text)

# x_text = np.concatenate(tmpdate,0)

def cov_to_class(val):
    return int(val/100)

# train_df = pd.DataFrame(tmpdate)
# print(train_df.shape)

# train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
from matplotlib import pyplot

#绘制直方图
def drawHist(tmpdate):
    x = np.array(tmpdate)
    # x = mu + sigma * np.random.randn(10000)

    num_bins = 50
    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, range=(0, 2000), normed=1, facecolor='blue', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, x.mean(), x.std())
    plt.plot(bins, y, 'r--')
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'Histogram of youliao news: $\mu=100$, $\sigma=15$')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

drawHist(tmpdate)

