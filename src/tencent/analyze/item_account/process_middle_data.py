# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import re
from src.conf import config


base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/'
# date_dir = '12_19'
date_dir = 'item_account'

result_dir = f"{base_dir}/{date_dir}"


debug_mode = False
def read_evaluate_file():
    file_name = os.path.join(base_dir, date_dir, f"test.csv")
    df = pd.read_csv(file_name, sep='\t')

    return df
    # das = df.to_json(orient='records')


def print_stat(df):
    # bb = df.to_dict(orient='records')
    # print(bb[len(bb) - 1])

    # 中位数、10%分位数、众数
    print(df['like_num'].median(), df['like_num'].quantile(0.1), df['like_num'].mode())
    like_num_df = df['like_num']
    play_num_df = df['play_num']
    expose_num_df = df['show_pv']


def get_url_df():
    url_file_name = os.path.join(base_dir, date_dir, "garlandliu_spkz_1605887750.csv")
    url_df = pd.read_csv(url_file_name, sep=',')
    url_df = url_df.rename(columns={'sShortVideoPlayUrl': 'url'})

    url_df.dropna(inplace=True)

    url_df['qbdocid'] = url_df['qbdocid'].apply(lambda x: int(re.sub('["=]', '', x)))
    url_df.drop_duplicates(subset=['qbdocid'], inplace=True)
    return url_df


def get_features_df():
    feature_file_name = os.path.join(base_dir, date_dir, "garlandliu_sptz_1605887917.csv")
    feature_df = pd.read_csv(feature_file_name, sep=',')

    feature_df.dropna(inplace=True)

    feature_df['qbdocid'] = feature_df['qbdocid'].apply(lambda x: int(re.sub('["=]', '', x)))
    feature_df = feature_df.drop(['原始docid'], axis=1)

    return feature_df


def join(df, url_df, feature_df):
    join_df = df.join(url_df.set_index('qbdocid'), on='video_id', how='inner')

    if 'url' in join_df.columns:
        join_df = join_df[~join_df['url'].str.contains('Android', na=False)]

    join_df = join_df.join(feature_df.set_index('qbdocid'), on='video_id', how='inner')

    final_df = join_df.reset_index(0)
    return final_df


def save_file(df, file_name):
    file_path = f'{result_dir}/{file_name}'
    df.to_csv(file_path)
    print(f'save to {file_path}')


def save_eval_result(file_name):
    file_name = f'{result_dir}/{file_name}'
    df = pd.read_csv(file_name, sep=',')
    df = df[df.eval('result >= 0')][['video_id', 'result']]
    videoI2result = dict(sorted(df.values.tolist()))

    result_file = os.path.join(base_dir, date_dir, f"result_{data_type}.pickle")
    result = {}
    if os.path.exists(result_file):
        with open(result_file, 'rb') as handle:
            result = pickle.load(handle)

    old_result_file = os.path.join(base_dir, date_dir, f"result_{data_type}_old.pickle")
    with open(old_result_file, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'copy old result to {old_result_file}')

    old_result_num = len(result)
    result.update(videoI2result)
    print(f'update result, old_num:{old_result_num} -> new_num:{len(result)}')

    with open(result_file, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'save result to {result_file}')
    print(result)


def print_dist_by_feature(df):
    print('-' * 20)
    post_confidence = 100
    c2_max_df = df[df.eval(f'c2_max >= {post_confidence}')]
    print(f'选取大于{int(post_confidence/10)}w消费作为置信度，一共:{len(c2_max_df[c2_max_df.eval("c2 == c2_max")])}条')

    data = []
    for col in cols:
        c2_max_df = c2_max_df[c2_max_df.eval('c2 == c2_max')]
        # print(f'{col}:')
        row = {'name': col}
        for i in np.arange(0.1, 1, 0.1):
            v = c2_max_df[col].quantile(i)
            # print(f'{int(i*100)}分位：{"%.4f" % c2_max_df[col].quantile(i)}')
            if col == 'floatClick':
                row[f'{int(i*100)}分位'] = str(int(v))
            else:
                row[f'{int(i * 100)}分位'] = f'{"%.2f" % (v * 100)}%'
        data.append(row)

    import pandas as pd
    from texttable import Texttable

    result_df = pd.DataFrame(data)
    tb = Texttable()
    tb.set_max_width(0)
    tb.set_cols_align(['l'] + ['l'] * 9)
    tb.set_cols_dtype(['t'] + ['t'] * 9)
    tb.header(result_df.columns)
    tb.add_rows(result_df.values, header=False)
    print(tb.draw())


if __name__ == '__main__':
    df = read_evaluate_file()

    columns = ['video_id', 'puin', 'finish_num', 'finish_percent', 'skip_5s_percent',
               'play_time', 'total_play_time', 'play_num', 'puin_name',
               'overall_level', 'metrics_7days_input',
               'metrics_7day_metrics_kv_storage', 'metrics_7days_st_kd',
               'kd_7day_total_exp_cnt', 'kd_7day_total_click_cnt',
               'liulanqi_fans_follow_cnt_week', 'kandian_fans_cnt',
               'liulanqi_fans_cnt', 'puin_active_status', 'f_create_time']

    df.dropna(inplace=True)

    # df = df.sample(n=10000)

    url_df = get_url_df()

    feature_df = get_features_df()

    df = join(df, url_df, feature_df)


    # df['url'] = df['video_id'].apply(lambda x: f'https://v.html5.qq.com/node/videoPlayer?vid={x}')
    # df.loc[:, 'url2'] = lambda x: f'https://v.html5.qq.com/node/videoPlayer?vid={x}'
    df['video_id'] = df['video_id'].apply(lambda x: f'_{x}')
    df.rename(columns={'sVideoName': 'title',
                       'floatClick': 'click',
                       'floatExposure': 'expose',
                       },
              inplace=True)
    df.eval('praise_percent = praiseNum/(expose+1.0)', inplace=True)
    df.eval('ctr = click/(expose+1.0)', inplace=True)
    df.eval('collect_percent = collectNum/(expose+1.0)', inplace=True)
    df.eval('share_percent = shareNum/(expose+1.0)', inplace=True)
    # df.eval('attention_percent = attention_num/(expose+1.0)', inplace=True) #todo：
    df.eval('comment_percent = commentNum/(expose+1.0)', inplace=True)
    df.eval('comment_percent = commentNum/(expose+1.0)', inplace=True)

    df.eval('play_time_vv = total_play_time/(play_num+1.0)', inplace=True)
    df.eval('liulanqi_fans_follow_cnt_week = liulanqi_fans_follow_cnt_week/(kd_7day_total_click_cnt+1.0)', inplace=True)

    # url title 真实曝光 消费 点赞 点赞率 收藏量 收藏率 分享量 分享率 关注量 关注率 写评论量 写评论率
    # 完播率 完播量 物理时长 单vv时长
    # -- 账号名 账号等级 账号7天发文量:metrics_7days_input
    # 账号7天发文量:metrics_7days_input
    # 账号7天启用量:metrics_7days_st_kd
    # 7天曝光量:kd_7day_total_exp_cnt
    # 七天消费量:kd_7day_total_click_cnt
    # 七天关注增长: liulanqi_fans_follow_cnt_week(浏览器周增粉丝数)
    # 七天转粉率（关注增量/消费量）kd_7day_follow_per_click
    # 账号粉丝数
    # 账号qq看点粉丝数
    # 账号末次发文时间: 活跃状态: 1-日活,2-周活,3-月活,4-90天活跃，5-流失。分别当日有发文，当周有发文，当月有发文，90天内发文，90天未发文,注意：统计最近90天活跃账号量：puin_active_status in (1,2,3,4)
    # 账号入驻时间:f_create_time
    select_columns = ['video_id', 'title', 'expose', 'click', 'praiseNum', 'ctr', 'praise_percent', 'collectNum', 'collect_percent', 'shareNum', 'share_percent', 'followSevenDay', 'commentNum', 'comment_percent',
                      'finish_num', 'finish_percent', 'play_time', 'play_time_vv',
                      'puin_name', 'overall_level', 'metrics_7days_input',
                      'metrics_7day_metrics_kv_storage', 'metrics_7days_st_kd',
                      'kd_7day_total_exp_cnt', 'kd_7day_total_click_cnt',
                      'liulanqi_fans_follow_cnt_week', 'liulanqi_fans_follow_cnt_week', 'liulanqi_fans_cnt', 'kandian_fans_cnt',
                      'puin_active_status', 'f_create_time', 'url']

    print(df[select_columns])
    save_file(df[select_columns], 'result.csv')



