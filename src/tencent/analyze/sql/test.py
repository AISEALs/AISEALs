import pandas as pd

df = pd.read_csv('优质原创物理时长-3609066-20211203203000.csv', sep='\t')

df = df.dropna()

sum_df = df[['mini_float_youzhi_num', 'mini_float_num']].sum()

df['mini_float_youzhi_num_percent'] = df['mini_float_youzhi_num']/sum_df.loc['mini_float_youzhi_num']
df['youzhi_percent'] = df['mini_float_youzhi_num_percent'].apply(lambda x: format(x, '.2%'))

df['mini_float_num_percent'] = df['mini_float_num']/sum_df.loc['mini_float_num']
df['mini_float_percent'] = df['mini_float_num_percent'].apply(lambda x: format(x, '.2%'))

df[['video_time_level', 'youzhi_percent', 'mini_float_percent']].to_csv('result.csv', sep='\t')
print(df[['video_time_level', 'youzhi_percent', 'mini_float_percent']])