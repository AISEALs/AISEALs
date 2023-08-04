# 非一线城市，实验桶vs对照桶 - 大发文曝光下降幅度
sql1 = '''
select
    tong,
    is_big_doc,
    count(1) as expose
from (
    SELECT
        CASE
            when (model_id like '%5081908%') then '5081908'
            when (model_id like '%5081907%') then '5081907'
            when (model_id like '%5081906%') then '5081906'
            when (model_id like '%5081905%') then '5081905'
            ELSE 'other'
        END as tong,
        case
            when cast(publish_doc_level as int) >= 1 then '1'
            else '0'
        END as is_big_doc
    from u_wsd.t_md_mtt_server_exposure_log
    where ds like '2021122210'
        and is_first_iter_city == '0'
        and publish_doc_level in ('0', '1', '2', '3', '4', '5')
    ) t1
where tong != 'other'
group by tong, is_big_doc
'''

# 城市-发文量级-曝光占比：
sql2 = """
SELECT
    is_first_iter_city,
    publish_doc_level,
    count(1) as expose
from u_wsd.t_md_mtt_server_exposure_log
where ds like '2021122211'
    and is_first_iter_city in ('0', '1')
    and publish_doc_level in ('0', '1', '2', '3', '4', '5')
GROUP by is_first_iter_city, publish_doc_level
"""

# 非一线城市+大发文，主题-曝光占比：
sql3 = """
SELECT
    is_first_iter_city,
    publish_doc_level,
    topic,
    count(1) as expose
from (
    select 
        is_first_iter_city,
        publish_doc_level,
        substring(subject, 0, 3) as topic
    from u_wsd.t_md_mtt_server_exposure_log
    where is_first_iter_city in ('0')
    and publish_doc_level in ('1', '2', '3', '4', '5')
) t
where ds like '2021122211'
GROUP by is_first_iter_city, publish_doc_level, topic
"""

# tong-大发文-高价值账号-物理时长-阅读时长 对比
sql4 = """
SELECT
    tong,
    is_big_doc,
    is_high_value_account,
    sum(video_time) as total_video_time,
    sum(play_time) as total_play_time,
    count(1) as total_expose
from (
    SELECT
        t1.guid,
        t1.video_id,
        t1.query_id,
        t1.puin_id,
        t1.tong,
        t1.is_big_doc,
        t1.is_high_value_account,
        t1.video_time,
        t2.play_time
    from (
        SELECT
            guid,
            video_id,
            query_id,
            puin_id,
            video_time,
            CASE
                when (gray_id like '%5307670%') then '5307670'
                when (gray_id like '%5307672%') then '5307672'
                when (gray_id like '%5307673%') then '5307673'
                ELSE 'other'
            END as tong,
            case
                when cast(publish_doc_level as int) >= 1 then '1'
                else '0'
            END as is_big_doc,
            case
                when cast(account_value_level as int) = 4 then '1'
                else '0'
            END as is_high_value_account
        from u_wsd.t_md_mtt_video_server_exposure_log
        where (ds >= 2021122715 and ds <= 2021122719)
            -- and scene = 1 and biz_type = 0
            -- and is_first_tier_city == '0'
            and publish_doc_level in ('0', '1', '2', '3', '4', '5')
            -- and scene = 1 and (biz_type = 0 or biz_type = 3)  -- feeds短视频
            and scene = 4  -- 小视频浮层
        ) t1 join u_wsd.t_md_mtt_video_playtime_log t2
            on t1.guid = t2.guid and t1.video_id = t2.video_id and t1.query_id = t2.query_id
        where t2.ds >= 2021122715 and t2.ds <= 2021122719 and t1.tong != 'other'
    ) t3
group by tong, is_big_doc, is_high_value_account
"""

# tong-大发文-高价值账号-后验 对比
sql = '''
SELECT
    tong,
    is_big_doc,
    is_high_value_account,
    sum(video_time) as total_video_time,
    sum(play_time) as total_play_time,
    count(1) as total_expose,
    sum(t3.favor) as favor,
    sum(t3.follow) as follow,
    sum(t3.share) as share,
    sum(t3.comment) as comment
FROM (
    SELECT
        guid,
        video_id,
        query_id,
        puin_id,
        case
            when cast(publish_doc_level as int) >= 1 then '1'
            else '0'
        END as is_big_doc,
        case
            when cast(account_value_level as int) = 4 then '1'
            else '0'
        END as is_high_value_account,
        CASE
            when (gray_id like '%5307670%') then '5307670'
            when (gray_id like '%5307672%') then '5307672'
            when (gray_id like '%5307673%') then '5307673'
            ELSE 'other'
        END as tong
    from u_wsd.t_md_mtt_video_server_exposure_log
    where ds like '20220103%'
        -- and scene = 1 and biz_type = 0
        -- and is_first_tier_city == '0'
        and publish_doc_level in ('0', '1', '2', '3', '4', '5')
        and scene = 4  -- 小视频浮层
) t1

LEFT OUTER JOIN
(
    SELECT
        guid, video_id, query_id,
        max(play_time) as play_time,
        max(video_time) as video_time,
        max(play_time)*1.0/max(video_time) as finish_rate,
        if(max(play_time) <= 3, 1, 0) as skip_rate
    FROM u_wsd.t_md_mtt_video_playtime_log
    WHERE biz_type in (0, 19) and ds like '20220103%' 
    GROUP BY guid, video_id, query_id
)t2 ON t1.guid = t2.guid and t1.query_id = t2.query_id and t1.video_id = t2.video_id

LEFT OUTER JOIN
(
    SELECT
        guid, 
        item_id as video_id,
        query_id,
        sum(if(action_type = 1, 1, 0)) as favor, 
        sum(if(action_type = 8, 1, 0)) as follow,
        sum(if(action_type = 7, 1, 0)) as share,
        sum(if(action_type = 0, 1, 0)) as comment
    from u_wsd.t_th_zixun_interact_log
    -- item_type：0-其他;1-视频;2-图文
    WHERE ds like '20220103%' AND item_type = 1
    GROUP BY query_id, guid, item_id
)t3 ON t1.guid = t3.guid and t1.video_id = t3.video_id and t1.query_id = t3.query_id
group by tong, is_big_doc, is_high_value_account
'''