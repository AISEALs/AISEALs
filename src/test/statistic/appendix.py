import numpy as np
import pandas as pd
from scipy import stats

# l = np.random.normal([5])
l = np.random.randint(low=-5, high=5, size=10)
m = np.mean(l)
abs_dispersion = [np.abs(m-x) for x in l]  # 多维矩阵下，只计算最外层维度的差值
MAD = np.mean(abs_dispersion)
MAD2 = pd.Series(l).mad()   # Mean Absolute Deviation

# aa = stats.median_absolute_deviation(l, scale=1)


print(np.var(l) /np.std(l))

low_values = [x for x in l if x < m]
semivar = np.sqrt(np.mean([(x-m)**2 for x in low_values]))

# import tushare as ts
#
# # 以铁矿石期货为例
# pro = ts.pro_api('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
# # 获取期货数据
# df = pro.fut_daily(ts_code='i.DCE', start_date='20180101', end_date='20190101', fields='trade_date,ts_code,close')
#
# # 用numpy计算最大回撤率
# max_drawdown = ((np.maximum.accumulate(df['close']) - df['close']) / np.maximum.accumulate(df['close'])).max()
# print('{:.2f}%'.format(max_drawdown))
#
# # 用pandas计算最大回撤率
# max_drawdown2 = ((df['close'].cummax() - df['close']) / df['close'].cummax()).max()
# print('{:.2f}%'.format(max_drawdown2))