import pandas as pd
import glob

file_names = r'./data/各层粉丝量级的作者量分布*.csv' # use your path
all_files = glob.glob(file_names)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    # df.drop(df.tail(2).index, inplace=True)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)
df = df[['日期', '粉丝量级', '内容真实曝光pv']]
#
print(df)
#
pdf = df.pivot(index='粉丝量级', columns='日期', values='内容真实曝光pv')

def sort_by_level(sl):
    if sl == '0':
        return -1
    aa = sl.split(',')
    aa = aa[0].strip('(')
    if aa.endswith('w'):
        aa = aa.replace('w', '0000')
    elif aa.endswith('k'):
        aa = aa.replace('k', '000')
    return int(aa)

pdf = pdf.reset_index()
pdf['row'] = pdf['粉丝量级'].apply(lambda x: sort_by_level(x))
pdf2 = pdf.sort_values('row')
# pdf2.drop(['row', '日期'])


columns = ['20211101~20211107', '20211201~20211207', '20220101~20220107', '20220201~20220207']
for col in columns:
    pdf2[col] = pdf2[col].apply(lambda x: float(x.replace(',', '',) if isinstance(x, str) else x))
    s = pdf2[col].astype(float).sum()
    pdf2[col] = pdf2[col].apply(lambda x: x/s).apply(lambda x: format(x, '.2%'))

print(pdf2)
pdf2.to_csv('result.csv', sep='\t')
