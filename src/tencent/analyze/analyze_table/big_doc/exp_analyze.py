import pandas as pd
import numpy as np


raw_df = pd.read_csv('data/统计后验数据-3710875-20220112102253.csv', sep='\t')

# df = df.pivot('tong', 'is_big_doc', 'expose').reset_index()
#
# df.columns = ['tong', 'bigdoc_false_expose', 'bigdoc_true_expose']
# df['total_expose'] = df['bigdoc_false_expose'] + df['bigdoc_true_expose']
# df['bigdoc_percent'] = df['bigdoc_true_expose']/df['total_expose']
#
# save_df = df[['tong', 'bigdoc_false_expose', 'total_expose', 'bigdoc_percent']]
# save_df.to_csv('result.csv', sep='\t')

pd.set_option('display.width', 1000) # 设置字符显示宽度

pd.set_option('display.max_rows', None) # 设置显示最大行

pd.set_option('display.max_columns', None) # 设置显示最大列

raw_df = raw_df[raw_df.tong == '5307672']
raw_df['is_exempt'] = raw_df.eval("is_big_doc == 1 and is_high_value_account == 1")

df = raw_df.copy()
df['avg_video_time'] = (df['total_video_time']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x))
df['avg_play_time'] = (df['total_play_time']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x))
df['avg_favor'] = (df['favor']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x*10000))
df['avg_follow'] = (df['follow']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x*10000))
df['avg_share'] = (df['share']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x*10000))
df['avg_comment'] = (df['comment']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x*10000))

df['expose'] = df['total_expose']
df['total_expose'] = df['expose'].sum()
df['percent'] = (df['expose']/df['total_expose']).apply(lambda x: format(x, '.2%'))

save_df = df[['tong', 'is_big_doc', 'is_high_value_account', 'avg_video_time', 'avg_play_time', 'avg_favor', 'avg_follow', 'avg_share', 'avg_comment', 'percent']]
print(save_df)
save_df.to_csv('result.csv', sep='\t')

df = raw_df.groupby(['tong', 'is_exempt'])[['total_video_time', 'total_play_time', 'total_expose', 'favor', 'follow', 'share', 'comment']].sum().reset_index()

df['avg_video_time'] = (df['total_video_time']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x))
df['avg_play_time'] = (df['total_play_time']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x))
df['avg_favor'] = (df['favor']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x*10000))
df['avg_follow'] = (df['follow']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x*10000))
df['avg_share'] = (df['share']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x*10000))
df['avg_comment'] = (df['comment']/df['total_expose']).apply(lambda x: '{:.2f}'.format(x*10000))

df['expose'] = df['total_expose']
df['total_expose'] = df['expose'].sum()
df['percent'] = (df['expose']/df['total_expose']).apply(lambda x: format(x, '.2%'))

save_df = df[['tong', 'is_exempt', 'avg_video_time', 'avg_play_time', 'avg_favor', 'avg_follow', 'avg_share', 'avg_comment', 'percent']]
print(save_df)


