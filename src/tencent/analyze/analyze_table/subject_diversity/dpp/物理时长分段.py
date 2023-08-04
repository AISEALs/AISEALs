import pandas as pd
from src.utils.analyze_tools import op_2dim_table

# df = pd.read_csv('first-subject-num.txt', sep='\t')
df = pd.read_csv('../data/playtime.csv', sep=',')
# df.columns = ['gray_id', 'subject', 'topic_num']#, 'video_time', 'play_time']
# print(df[df[['gray_id', 'subject']].duplicated()])
# df.drop_duplicates(subset=['gray_id', 'subject'], inplace=True)
df.fillna(0, inplace=True)

# df['subject'] = df['subject'].astype(int)
# df['play_time'] = df['play_time'].astype(int)
# df['video_time'] = df['video_time'].astype(int)


target = 'playtime'
# target = 'play_time'
pdf = df.pivot(index='time_int', columns='gray', values=target)
pdf.fillna(0, inplace=True)
print(pdf)
pdf.sort_values(by='time_int', ascending=True, inplace=True)
pdf['c'] = ((pdf[9363271] - pdf[9363270])/pdf[9363270]).apply(lambda x: format(x, '.2%'))
# pdf['c2'] = ((pdf[8698819] - pdf[8698820])/pdf[8698820]).apply(lambda x: format(x, '.2%'))


# columns = list(range(7626522, 7626525+1))
columns = [9363270, 9363271]
# print(columns)
pdf = op_2dim_table.cal_percent_by_cols(pdf, columns=columns, trans_col_func=None, save=True)
print(pdf)


