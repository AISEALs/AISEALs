
# 抽取play_time
sql = '''
SELECT
    play_time_str,
    substr(play_time_str, 2, length(play_time_str) - 2) as play_time
from (
    select
        regexp_extract(ext_info, 'playtime(.*?)recallStrategy', 1) as play_time_str
    from u_wsd.t_md_mtt_report_predict_float_sample
    where ds = 2021120100
    limit 10
) t
'''

# 小视频浮层，短小视频点击率
sql2 = """
SELECT
    gray_id,
    busi,
    label,
    count(*) as num
from (
    SELECT
        gray_id,
        busi,
        cast(substr(play_time_str, 2, length(play_time_str) - 2) as int) as play_time,
        label
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
            END as busi,
            regexp_extract(ext_info, 'playtime(.*?)recallStrategy', 1) as play_time_str,
            label
        FROM u_wsd.t_md_mtt_report_predict_float_sample
        where ds like '20211201%' and biz_type = 2
            -- and ext_info like '%iFollow:1%'
        ) t1
    where gray_id != 'other' and busi != 'no'
    ) t2
GROUP by gray_id, busi,label;
"""