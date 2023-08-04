import pandas as pd
from src.utils.analyze_tools import op_2dim_table

# df = pd.read_csv('first-subject-num.txt', sep='\t')
df = pd.read_csv('../data/0425-sub-finishrate2.csv', sep=',')
df.columns = ['gray_id', 'subject', 'video_time', 'play_time', 'finish_rate']
print(df[df[['gray_id', 'subject']].duplicated()])
df.drop_duplicates(subset=['gray_id', 'subject'], inplace=True)
df.fillna(0, inplace=True)

# df['subject'] = df['subject'].astype(int)
df['play_time'] = df['play_time'].astype(int)
df['video_time'] = df['video_time'].astype(int)
df['finish_rate'] = df['finish_rate'].astype(float)


target = 'finish_rate'
# target = 'play_time'
pdf = df.pivot(index='subject', columns='gray_id', values=target)
pdf.fillna(0, inplace=True)
print(pdf)
pdf.sort_values(by='subject', ascending=True, inplace=True)


# columns = list(range(7626522, 7626525+1))
columns = [7781783, 7781784]
print(columns)
# pdf = op_2dim_table.cal_percent_by_cols(pdf, columns=columns, trans_col_func=None, save=True)

for col in columns:
    pdf[col] = pdf[col].apply(lambda x: format(x, '.2f'))

pdf[columns].to_csv('result.csv', sep='\t')


