# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import re
from src.conf import config


base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/'
date_dir = 'item_feature'

result_dir = f"{base_dir}/{date_dir}"

# scene = 'doc'
# scene = 'video'
scene = 'video_banyun'

if scene == 'doc':
    item_column = 'doc_id'
else:
    item_column = 'video_id'

import re
def remove_prefix(text, prefix):
    return re.sub(r'^{0}'.format(re.escape(prefix)), '', text)

debug_mode = False
def read_raw_feature():
    if scene == 'doc':
        name = "随机图文item-3115660-20210602170338.csv"
    elif scene == 'video':
        name = "随机视频item-0604.csv"
    elif scene == 'video_banyun':
        name = "随机视频搬运号item-0604.csv"

    file_name = os.path.join(base_dir, date_dir, name)
    # sep = '\t' if scene == 'doc' else ','
    sep = '\t'
    df = pd.read_csv(file_name, sep=sep)

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
    if scene == 'doc':
        name = 'garlandliu_twkz_1622624771.csv'
    elif scene == 'video':
        name = 'garlandliu_spkz_1623054091.csv'
    else:
        name = 'garlandliu_spkz_1623054879.csv'
    url_file_name = os.path.join(base_dir, date_dir, name)
    url_df = pd.read_csv(url_file_name, sep=',')
    url_df = url_df.rename(columns={'sShortVideoPlayUrl': 'url'})
    url_df = url_df.rename(columns={'sUrl': 'url'})

    # url_df.dropna(inplace=True)

    url_df['qbdocid'] = url_df['qbdocid'].apply(lambda x: int(re.sub('["=]', '', x)))
    url_df.drop_duplicates(subset=['qbdocid'], inplace=True)
    return url_df


def get_features_df():
    if scene == 'doc':
        name = 'garlandliu_twtz_1622625207.csv'
    elif scene == 'video':
        name = 'garlandliu_sptz_1623054283.csv'
    else:
        name = 'garlandliu_sptz_1623055409.csv'

    feature_file_name = os.path.join(base_dir, date_dir, name)
    feature_df = pd.read_csv(feature_file_name, sep=',')

    # feature_df.dropna(inplace=True)

    feature_df['qbdocid'] = feature_df['qbdocid'].apply(lambda x: int(re.sub('["=]', '', x)))
    feature_df = feature_df.drop(['原始docid'], axis=1)

    return feature_df


def join(df, url_df, feature_df=None):
    join_df = df.join(url_df.set_index('qbdocid'), on=item_column, how='inner')

    if 'url' in join_df.columns:
        join_df = join_df[~join_df['url'].str.contains('Android', na=False)]

    if feature_df is not None:
        join_df = join_df.join(feature_df.set_index('qbdocid'), on=item_column, how='inner')

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
    df = read_raw_feature()

    print(df.columns)

    # df = df.sample(n=10000)

    url_df = get_url_df()

    feature_df = get_features_df()

    df = join(df, url_df, feature_df)
    print(df.columns)

    # df.dropna(inplace=True)
    df = df.sample(n=500)


    if scene == 'doc':
        df.eval('praise_percent = praiseNum/(exposureTotal+1.0)', inplace=True)
        df.eval('ctr = clickTotal/(exposureTotal +1.0)', inplace=True)
        df.eval('share_percent = shareNum/(exposureTotal+1.0)', inplace=True)
        df.eval('comment_percent = commentNum/(exposureTotal +1.0)', inplace=True)
        df.eval('meanReadTime = readTotalTime /(readTotalNum +1.0)', inplace=True)
    else:
        df.eval('ctr = floatClick/(floatExposure +1.0)', inplace=True)
        df.eval('floatPlayTotalTime = floatPlayTotalTime /(floatVideoTotalTime +1.0)', inplace=True)

    if scene == 'doc':
        select_columns = ['doc_id', 'puin_id', 'overall_level', 'metrics_7days_input',
          'metrics_7days_st_kd', 'accounttype', 'puin_qb_7day_follow_rate',
          'exposureTotal', 'clickTotal', 'ctr', 'commentNum', 'comment_percent',
          'shareNum', 'share_percent', 'praiseNum', 'praise_percent',
          'meanReadTime', 'readTotalTime', 'readTotalNum', 'url']
    else:
        select_columns = ['video_id', 'scene', 'puin', 'puin_name', 'overall_level', 'metrics_7days_input',
           'metrics_7days_st_kd', 'accounttype', 'puin_qb_7day_follow_rate',
           'follow_rate_rank', 'floatExposure', 'feedsExposure', 'floatClick', 'feedsClick', 'ctr',
           'floatPlayTotalTime', 'floatVideoTotalTime', 'commentNum', 'shareNum',
           'praiseNum', 'followSevenDay', 'url']
    print(df[select_columns])
    save_file(df[select_columns], 'result.csv')
