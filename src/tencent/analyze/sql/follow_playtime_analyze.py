import pandas as pd
import numpy as np


df = pd.read_csv('关注-时长分析.csv', sep='\t')
print(df.groupby('busi')['play_time'].describe())
