
sql = '''
### biz_type=2 
SELECT
    gray_id,
    busi,
    count(*) as follow_num
from (
    SELECT
        CASE
            when ext_info like '%4022987%' then '4022987'
            when ext_info like '%4022984%' then '4022984'
            when ext_info like '%4022980%' then '4022980'
            else 'other'
        END as gray_id,
        case
            when ext_info like '%isShortInMiniVideo:1%' then 'short'
            when ext_info like '%isShortInMiniVideo:0%' then 'mini'
            else 'no'
        END as busi
    FROM u_wsd.t_md_mtt_report_predict_float_sample
    where ds like '20211201%' and biz_type = 2
        and ext_info like '%iFollow:1%'
    ) t
where gray_id != 'other' and 'busi' != 'no'
GROUP by gray_id, busi;
'''

import pandas as pd

# df = pd.read_csv('短小视频关注差异-3606036-20211203120804.csv', sep='\t')
# df = pd.read_csv('短小视频关注差异-3606241-20211203125059.csv', sep='\t')
# df['percent'] = df['follow_num']/df['play_time_num']
#
# df[['busi', 'follow_num', 'play_time_num', 'percent']].to_csv('test.csv', sep='\t')
# print(df)
df = pd.read_csv('短小视频关注差异-3608179-20211203191233.csv', sep='\t')
df.columns = [col.replace('t3.', '') for col in df.columns]
print(df.columns)

df2 = df[df.total_num > 5000].pivot(index='doc_id', columns='busi', values=['follow_num', 'play_time_num', 'run_time']).dropna()

# doc_ids = set(df2.reset_index().doc_id.to_list())
df2 = df2.reset_index()

df3 = df2['follow_num']/df2['play_time_num']