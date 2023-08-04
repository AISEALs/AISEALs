import pandas as pd


def value(item):
    return item[item.find(':') + 1:]


col_idx = [3, 4, 5, 6, 7]
columns = 'bStrategyIdV2Gray,bESExpSwitch,grayid,isFilter,stg_ids'.split(',')
grayid = '8379889'
# grayid = '8379890'
# grayid = 'all'
file_name = f'./stg_test.log.{grayid}'
df = pd.read_table(file_name, header=None, delimiter='|',
                   usecols=col_idx,
                   converters={i: value for i in col_idx},
                   names=columns)
df['bESExpSwitch'] = df.bESExpSwitch.astype(int)
df['isFilter'] = df.isFilter.astype(int)
print(df.grayid.value_counts())
if grayid == 'all':
    df = df[df.bESExpSwitch == 0]

df['stg_id_list'] = df.stg_ids.str.split(',').apply(lambda l: list(filter(None, l)))
df2 = df.explode('stg_id_list', ignore_index=True)
df2.rename(columns={'stg_id_list': 'stg_id'}, inplace=True)
df2.dropna(inplace=True)

total_filter_percent = format(len(df[df.isFilter == 1])/len(df), '.2%')
print(f'{grayid} total filter percent: {total_filter_percent}')
# print(f'total query num: {len(df)}, total stg sample len: {len(df2)}')


def filter_num(l):
    return sum(l)


# stg_ids = [54, 2, 212, 65]
df4 = df2.groupby('stg_id')['isFilter'].agg(['count', filter_num]).reset_index()
df4['stg_id'] = df4.stg_id.astype(int)
df4['filter_percent'] = df4.eval('filter_num/count').apply(lambda x: format(x, '.2%'))
df5 = df4.sort_values(by='count', ascending=False)
print(df5)
df5.to_csv('result.csv', sep='\t')