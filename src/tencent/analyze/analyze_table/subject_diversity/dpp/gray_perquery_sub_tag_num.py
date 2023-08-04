import pandas as pd
from src.utils.analyze_tools import op_2dim_table

# df = pd.read_csv('first-subject-num.txt', sep='\t')
df = pd.read_csv('../data/gray_query_subtag_num.csv', sep=',')

df2 = df.set_index('grayid').T
df2[range(6713452, 6713456)].round(2).to_csv('result.csv', sep='\t')