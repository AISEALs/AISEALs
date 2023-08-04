
import pandas as pd

from io import StringIO
str = """
2021120712	小视频	265941
2021120712	视频	67417
2021120812	小视频	269993
2021120812	视频	68428
2021120912	视频	71885
2021120912	小视频	275844
2021121012	视频	70059
2021121012	小视频	273049
"""
data = StringIO(str)
df = pd.read_csv(data, sep='\t', header=None)

# df = pd.read_csv("/Users/jiananliu/work/projects/data/analyze/table/sample100/RERANK-短视频占比.txt", sep='\t')
print(df)

df.columns = ['dateHour', 'busi', 'expose']
# df['dateDay'] = df['dateHour'].apply(lambda x: int(x/100))
df = df.groupby(['dateHour', 'busi'])['expose'].agg(sum).reset_index()

df = df.pivot(index='dateHour', columns='busi', values='expose')

df['short_percent'] = df['视频']/(df['视频'] + df['小视频'])

print(df)
# df.sort_index().to_csv('short_percent.csv')
