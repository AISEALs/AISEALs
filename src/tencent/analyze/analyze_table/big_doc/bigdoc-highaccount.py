import numpy as np
import pandas as pd

df = pd.read_csv("./data/实验对比-3674912-20211227191433.csv", sep='\t')

col = 'puin_num'
df['tong_bigdoc_expose'] = df.groupby(['tong', 'is_big_doc'])[col].transform(np.sum)
df['tong_expose'] = df.groupby(['tong'])[col].transform(np.sum)
df['tong_percent'] = (df[col]/df['tong_bigdoc_expose']).apply(lambda x: format(x, '.2%'))
df['big_doc_percent'] = (df['tong_bigdoc_expose']/df['tong_expose']).apply(lambda x: format(x, '.2%'))

print(df[['tong', 'is_big_doc', 'is_high_value_account', 'tong_percent', 'big_doc_percent']])
# print(df[['tong', 'is_big_doc', 'is_high_value_account', 'big_doc_percent']])
