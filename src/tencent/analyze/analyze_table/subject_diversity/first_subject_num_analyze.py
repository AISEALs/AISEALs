import pandas as pd

# df = pd.read_csv('first-subject-num.txt', sep='\t')
df = pd.read_csv('data/feeds-video-subject-nums.txt', sep='\t')
df.columns = ['server_name', 'subject', 'topic_num']
df.fillna(0, inplace=True)
df['subject'] = df['subject'].astype(int)

pdf = df.pivot(index='subject', columns='server_name', values='topic_num')
pdf.fillna(0, inplace=True)
pdf.sort_values(by='RECALL', ascending=False, inplace=True)

for col in pdf.columns:
    print(col)
    col_pdf = pdf[col]
    sum_num = col_pdf.sum()
    pdf[col + '_percent'] = (col_pdf/sum_num).apply(lambda x: format(x, '.2%'))

pdf = pdf[['RECALL', 'RECALL_percent', 'RANK', 'RANK_percent', 'PREDICT', 'PREDICT_percent', 'RERANK', 'RERANK_percent']]

pdf.to_csv('result.csv', sep='\t')
