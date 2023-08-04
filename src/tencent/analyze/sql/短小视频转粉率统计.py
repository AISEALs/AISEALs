
sql = '''
SELECT
    gray_id,
    busi,
    sum(if (follow = 'follow_1', 1, 0)) as follow_num,
    sum(if ((follow = 'follow_1' and play_time > 0), 1, 0)) as follow_playtime0_num,
    sum(if (play_time > 0, 1, 0)) as play_time_num,
    count(*) as totoal_num
from (
    SELECT
        gray_id,
        busi,
        follow,
        cast(substr(play_time_str, 2, length(play_time_str) - 2) as int) as play_time,
        label
    from (
        SELECT
            CASE
                when ext_info like '%2673394%' then '2673394'
                else 'all'
            END as gray_id,
            case
                when (biz_type == 2 and ext_info like '%isShortInMiniVideo:1%') then 'miniscene_short'
                when (biz_type == 2 and ext_info like '%isShortInMiniVideo:0%') then 'miniscene_mini'
                when (biz_type == 1) then 'shortscene_short'
                else 'no'
            END as busi,
            case
                when (ext_info like '%iFollow:1%') then 'follow_1'
                when (ext_info like '%iFollow:0%') then 'follow_0'
                else 'other'
            END as follow,
            regexp_extract(ext_info, 'playtime(.*?)recallStrategy', 1) as play_time_str,
            label,
            biz_type
        FROM u_wsd.t_md_mtt_report_predict_float_sample
        where ds like '20211223%' and biz_type = 2
            
        ) t1
    where gray_id != 'other' and busi != 'no' and biz_type in (1, 2)
    ) t2
-- where play_time > 0
GROUP by gray_id, busi
'''