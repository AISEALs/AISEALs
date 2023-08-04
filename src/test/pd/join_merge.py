import pandas as pd

# df1 = pd.DataFrame({'lkey': ['A', 'B', 'C'],
#                     'value': [1, 2, 3]})
#
# df2 = pd.DataFrame({'rkey': ['A', 'B', 'D'],
#                     'value': [1, 2, 4]})
#
# df1.merge(df2, left_on='lkey', right_on='rkey')
# df1[['lkey']].merge(df2[['rkey']], left_on='lkey', right_on='rkey', how='outer')

df1 = pd.DataFrame({'key': ['A', 'B', 'C'],
                    'value': [1, 2, 3]})

df2 = pd.DataFrame({'key': ['A', 'B', 'D'],
                    'value': [1, 2, 4]})
df1[['key']].merge(df2[['key']], left_on='key', right_on='key', how='outer')
df1[['key']].set_index('key').join(df2[['key']].set_index('key'), how='outer')
