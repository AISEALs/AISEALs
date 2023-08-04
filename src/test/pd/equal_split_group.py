import pandas as pd
import numpy as np

# 模拟数据
data = pd.DataFrame({'price': np.random.randn(1000),
                     'amount': 100*np.random.randn(1000)})

# 等分价格为10个区间
quartiles = pd.cut(data.price, 10)

# 定义聚合函数
def get_stats(group):
    return {'s': group.sum()}

# 分组统计
grouped = data.amount.groupby(quartiles)
price_bucket_amount = grouped.apply(get_stats).unstack()

print(price_bucket_amount)