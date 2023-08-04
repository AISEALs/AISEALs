import pandas
import pandas as pd

df = pd.read_csv('test.log', sep='|', header=None)

df = df[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19]]

df.columns = ['gray_id', 'praise', 'share', 'rcomment', 'wcomment', 'follow', 'feedback', 'commenttime', 'homepage', 'mergescore', 'user_vec', 'item_vec']

df['user_vec'] = df['user_vec'].apply(lambda x: x.split(':')[1])
df['item_vec'] = df['item_vec'].apply(lambda x: x.split(':')[1])

df['user_vec_list'] = df['user_vec'].str.split(',')
df['len'] = df['user_vec_list'].transform(len)
df = df[df.len >= 48]


for i in range(48):
    print(f'-------index:{i}---------')
    print(df['user_vec_list'].apply(lambda x: x[i]).astype(float).describe())


# df1 = df[df['gray_id'] == 'all']
# df2 = df[df['gray_id'] == '3399385']
print(df['user_vec_list'].apply(lambda x: x[0]).astype(float).describe())