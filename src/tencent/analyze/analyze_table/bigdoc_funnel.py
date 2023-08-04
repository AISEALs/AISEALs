import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd

df = pd.read_csv('大发文漏斗-feeds.txt', sep='\t')

df.columns = ['server_name', 'publish_level', 'puin_characteristic', 'expose']

print(df)

df['total_expose'] = df.groupby('server_name')['expose'].transform('sum')

bigdf = df[df.puin_characteristic == '1']
bigdf = bigdf.pivot(index='server_name', columns='publish_level', values=['expose', 'total_expose'])

bigdf.columns = bigdf.columns.to_flat_index().map('_'.join)
bigdf['total_expose'] = bigdf['total_expose_10+']

bigdf['precent10'] = bigdf['expose_10+']/bigdf['total_expose']
print(bigdf[['expose_10+', 'total_expose', 'precent10']])


# TESTDATA = StringIO(puin2name)

# puin2name_df = pd.read_csv(TESTDATA, sep="\t")

# df = pd.read_csv('subject_percent.csv', sep='\t')
#
# # df = df.merge(puin2name_df, on='puin')
#
# df['total_expose'] = df.groupby('puin')['expose'].transform('sum')
# df['expose_percent'] = df['expose']/df['total_expose']
# df['expose_percent'] = df['expose_percent'].apply(lambda x: format(x, '.2%'))
#
# df['total_doc_num'] = df.groupby('puin')['doc_num'].transform('sum')
# df['st_doc_percent'] = df['doc_num']/df['total_doc_num']
# df['st_doc_percent'] = df['st_doc_percent'].apply(lambda x: format(x, '.2%'))
#
# df2 = pd.read_csv('subject_percent2.csv', sep='\t')
# df2.columns = ['puin', 'puin_name', 'fst_chann_cn', 'send_doc_num']
# df2['send_total_doc_num'] = df2.groupby('puin')['send_doc_num'].transform('sum')
# df2['send_doc_percent'] = df2['send_doc_num']/df2['send_total_doc_num']
# df2['send_doc_percent'] = df2['send_doc_percent'].apply(lambda x: format(x, '.2%'))
# df2['total_send_doc_num'] = df2['send_total_doc_num']
#
# # df = df2.merge(df, left_on=['puin', 'fst_chann_cn'], right_on=['puin', 'fst_chann_cn'], how='left')
# print(df)
# print(df.to_csv('result.csv'))
# # df[['puin', 'puin_name', 'fst_chann_cn', 'send_doc_num', 'total_send_doc_num', 'send_doc_percent', 'doc_num', 'total_doc_num', 'st_doc_percent', 'expose', 'total_expose', 'expose_percent']].to_csv('result.csv')
# df2[['puin', 'puin_name', 'fst_chann_cn', 'send_doc_num', 'send_total_doc_num', 'send_doc_percent']].to_csv('result.csv')
