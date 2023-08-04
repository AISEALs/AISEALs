# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import re
from src.conf import config


base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/'
# date_dir = '12_19'
date_dir = 'short_time'
# data_type = '50w'
data_type = '7w'

result_dir = f"{base_dir}/result/"


debug_mode = False
def read_evaluate_file():
    file_name = os.path.join(base_dir, date_dir, f"item-eval-{data_type}.csv")
    df = pd.read_csv(file_name, sep='\t')

    columns = ['video_id', 'show_pv', 'comment_num', 'like_num', 'dislike_num',
               'comment_1_num', 'comment_0_num', 'comment_look_num',
               'not_interest_num', 'share_num', 'attention_num', #'report_num',
               'total_play_time', 'total_video_time', 'video_num', 'play_num',
               'finish_90_watch_num', 'finish_read_90_percent', 'skip_read_10_percent',
               'history_comment_cnt', 'history_share_cnt', 'history_favor_cnt',
               'history_collect_cnt', 'history_feedback_cnt', 'history_quality_score',
               'guid', 'puin_id', 'account_id', 'account_score', 'account_ctr',
               'account_level', 'playtimelevel_playcnt', 'account_expose',
               'account_click', 'cover_pic_score', 'content_clarity', 'new_good_cp']

    if debug_mode:
        select_columns1 = ['video_id', 'finish_read_90_percent', 'skip_read_10_percent',
                       'show_pv', 'play_num']
    else:
        select_columns1 = ['video_id', 'finish_read_90_percent', 'skip_read_10_percent',
                           'history_comment_cnt', 'history_share_cnt', 'history_favor_cnt',
                           'show_pv', 'play_num']

    return df[select_columns1]
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
    url_file_name = os.path.join(base_dir, date_dir, "garlandliu_sptz_1605024076.csv")
    url_df = pd.read_csv(url_file_name, sep=',')
    url_df = url_df.rename(columns={'sShortVideoPlayUrl': 'url'})

    url_df['qbdocid'] = url_df['qbdocid'].apply(lambda x: int(re.sub('["=]', '', x)))
    url_df.drop_duplicates(subset=['qbdocid'], inplace=True)
    return url_df


def get_features_df():
    feature_file_name = os.path.join(base_dir, date_dir, "feature.csv")
    feature_df = pd.read_csv(feature_file_name, sep='\t')

    select_columns = ['commentNum', 'shareNum', 'praiseNum', 'collectNum',
                      'feedbackNum', 'seeingCommentNum', 'mainExposure',
                      'mainClick', 'floatExposure', 'floatClick']

    # return feature_df[select_columns]
    return feature_df


def join(df, url_df, feature_df):
    join_df = df.join(url_df.set_index('qbdocid'), on='video_id', how='inner')

    if 'url' in join_df.columns:
        join_df = join_df[~join_df['url'].str.contains('Android', na=False)]

    join_df = join_df.join(feature_df.set_index('video_id'), on='video_id', how='inner')

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
    # df = read_evaluate_file()
    from src.tencent.analyze.upgrade_10w_time import process_middle_data
    df = process_middle_data.read_middle_data()

    scene_id = 3

    df = df[df.eval(f'scene == {scene_id}')]

    url_df = get_url_df()
    feature_df = get_features_df()
    df = join(df, url_df, feature_df)

    df['url'] = df['video_id'].apply(lambda x: f'https://v.html5.qq.com/node/videoPlayer?vid={x}')
    # df.loc[:, 'url2'] = lambda x: f'https://v.html5.qq.com/node/videoPlayer?vid={x}'
    df['video_id'] = df['video_id'].apply(lambda x: f'_{x}')
    df.eval('praise_percent = praiseNum/floatExposure', inplace=True)
    df.eval('ctr = floatClick/floatExposure', inplace=True)
    df.eval('collect_percent = collectNum/floatExposure', inplace=True)
    df.eval('share_percent = shareNum/floatExposure', inplace=True)
    df.eval('attention_percent = attention_num/floatExposure', inplace=True) #todo：
    df.eval('comment_percent = commentNum/floatExposure', inplace=True)
    df.eval('see_comment_percent = seeingCommentNum/floatExposure', inplace=True)



    reach_10w_num_range = [0, 1, 5] + list(range(10, 120, 10))
    # video_c2_df = df.pivot(index='video_id', columns='c2')[reach_10w_num_range]

    analyze_range = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 10000)]
    cols = ['floatClick', 'finish_read_90_percent', 'skip_read_10_percent', 'praise_percent', 'praiseNum', 'ctr', 'collect_percent', 'collectNum', 'share_percent', 'shareNum', 'attention_percent', 'attention_num', 'comment_percent', 'commentNum', 'see_comment_percent', 'seeingCommentNum']
    # for i in range(0, 110, 10):
    for (start, end) in analyze_range:
        c2_max_df = df[df.eval(f'c2_max >= {start} and c2_max < {end}')]
        c2_max_df = c2_max_df[c2_max_df.eval('c2 == c2_max')]
        print('-' * 20)
        print(f'消费达到：{int(start/10)}w-{int(end/10)}w, count:{c2_max_df.video_id.count()}')
        s50_like_percent = c2_max_df[cols].quantile(0.5)
        print(f'50分位值:\n{s50_like_percent}')

    # -------------------
    print_dist_by_feature(df)

    df2 = df[df.eval('c2 == c2_max')]
    df2 = df2.reset_index(0)
    df2 = df2[['video_id'] + cols + ['url']]
    save_file(df2, 'short_3day_result.csv')


