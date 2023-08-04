import numpy as np
import pandas as pd

file_name = '/Users/jiananliu/work/analyze/data/dpp/5953190.log'
df = pd.read_csv(file_name, header=None, sep='|')

df = df[list(range(3, 18))]
df.head(10)

df.columns = ['queryid', 'grayid', 'lastdocid', 'docid', 'sim', 'dpp_alpha', 'dpp_bata',
              'use_raw_mode', 'docInfo.vtDppCiSize', 'Lij', 'ei',
              'calScore', 'docInfoDScore', 'sourcedDppScore', 'docInfo.dScore']

for col in df.columns:
    df[col] = df[col].apply(lambda x: x.split(':')[1])

# max_len = len("queryid:04c29279be7641c4dccbb4a407c488cb_1642624571945")
# idx = df['queryid'].apply(lambda x: len(x) <= max_len)

print(df)
def dpp_result_by_queryid(queryid: str):
    df2 = df[df['queryid'] == queryid]

    # df2[['queryid', 'lastdocid', 'docid']].groupby(['queryid', 'lastdocid']).count()
    df2[['queryid', 'lastdocid', 'docid']].groupby('queryid').nunique()
    print(df2[df2.docid == '2606524445269945786'][['lastdocid', 'docid', 'docInfoDScore', 'docInfo.dScore']])

    # np.argmax(df2.groupby('lastdocid')['docInfo.dScore'])
    idx = df2.groupby('lastdocid')['docInfo.dScore'].transform(max) == df2['docInfo.dScore']
    print(df2[idx][['lastdocid', 'docid', 'docInfoDScore', 'docInfo.dScore']])


queryid = 'f036668d75d4ff7e29185f5013b788cb_1642645454568'
dpp_result_by_queryid(queryid)
