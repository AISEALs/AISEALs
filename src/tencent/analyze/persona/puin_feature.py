import os
import argparse
import sys
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

sys.path.append('')


# 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='BigPublishDoc-Profile-v2')
    # scene = 'mini'  # 视频
    # scene = 'short'  # 视频
    # scene = 'feeds' # 图文
    parser.add_argument('--scene', type=str, default='mini')
    parser.add_argument('--new_version', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--base_dir', type=str,
                        default='/Users/jiananliu/work/analyze/data/persona',
                        help='please set')
    parser.add_argument('--sub_dir_name', type=str,
                        default='puin_feature')
    parser.add_argument('--use_multi_proc', default=False, action='store_true',
                        help='is use multiprocessing lib to add speed')
    parser.add_argument('--multi_proc_num', type=int,
                        default=10)
    parser.add_argument('--show_table', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_pic', type=str,
                        default='expose_trend.png')
    parser.add_argument('--date', type=str,
                        default='20210411')
    parser.add_argument('--save_path', type=str,
                        default='result.csv')
    parser.add_argument('--task_type', type=int, required=False, default=3,
                        help="0: run middle data, gen result.csv.\n"
                             "1: run middle data, gen result.csv.\n"
                             "2: read result.csv to gen trend pic\n")

    return parser.parse_args()


def cal_alpha(expose):
    # def sigmoid(x):
    #     if x > 0:
    #         return 1 / (1 + math.exp(-x))
    #     else:
    #         return 1 - 1 / (1 + math.exp(x))
    # todo: replace by sigmoid
    ret = 0.7 if expose > 10000 else 0.3
    return ret


def filter(df):
    df2 = df[df.expose > 3000]
    # overall_level
    print(df['overall_level'].value_counts())
    df3 = df2[df2['overall_level'] != 4.0]
    # print(f"剩下占比：{len(df3)/len(df2)}")
    # metrics_characteristic_account_tag 1=搬运号；2=原创号；3=拆条号 4=人设号
    print(df['metrics_characteristic_account_tag'].value_counts())
    df4 = df3[df3['metrics_characteristic_account_tag'] == '1']
    # cms_account_copyright 版权号标识（银河侧）:0-不是版权号，1，2...是版权号
    df5 = df4[df4['cms_account_copyright'] != 1]
    # is_75_low_standard_mark_id
    return df5


if __name__ == '__main__':
    args = get_args()

    file_name = os.path.join(args.base_dir, args.sub_dir_name, "20210615.gz")
    columns = ['puin', 'con_1_day', 'con_7_day', 'con_14_day', 'con_30_day',
        'expose_1day',
        'click_1day', 'favor_1day', 'follow_1day', 'share_1day', 'comment_1day',
        'expose',
        'click', 'comment', 'share', 'favor', 'collect', 'feedback',
        'kandian_fans_cnt', 'follow', 'liulanqi_fans_follow_cnt_week', 'jubao_rate', 'dizhi_75_rate',
        'cover_pic_score', 'content_clarity', 'overall_level',
        'cc_ai_account_professional_score', 'metrics_characteristic_account_tag', 'cms_account_copyright', 'is_75_low_standard_mark_id', 'follow_rate']

    df = pd.read_csv(file_name, sep='|', compression='gzip')
    df.columns = columns

    df['cc_ai_account_professional_score'] = pd.to_numeric(df['cc_ai_account_professional_score'], errors='coerce').astype(np.float)
    df['cms_account_copyright'] = pd.to_numeric(df['cms_account_copyright'], errors='coerce').astype(np.float)
    df['overall_level'] = pd.to_numeric(df['overall_level'], errors='coerce').astype(np.float)
    df['metrics_characteristic_account_tag'] = df.metrics_characteristic_account_tag.astype(str)
    df['cms_account_copyright'] = df.cms_account_copyright.astype(str)
    df['follow'] = pd.to_numeric(df['follow'], errors='coerce').astype(np.float).fillna(0).astype(int)

    print(df.metrics_characteristic_account_tag.value_counts())
    print(df.cms_account_copyright.value_counts())

    df[['con_1_day', 'con_7_day', 'con_14_day']].describe()

    # con_14_day_p99 = df.con_14_day.quantile(0.99)
    # df2 = df[df['con_14_day'] > con_14_day_p99]
    # is_75_low_standard_mark_id
    df['low_standard_75_num'] = df['is_75_low_standard_mark_id'].astype(str).apply(lambda x: len(x.strip('&').split('&')) if x else 0)

    df2 = df.copy()
    q = 0.5
    for feat in ['click']:
        col_percent = feat + '_percent'
        df[col_percent] = df.eval(f"{feat}/(expose + 1)")
        col_percent_norm = col_percent + '_norm'
        df[col_percent_norm] = MinMaxScaler().fit_transform([[i] for i in df[feat].values]).flatten()

    posterior_col = ['comment', 'share', 'favor']
    for feat in posterior_col:
        col_percent = feat + '_percent'
        df[col_percent] = df.eval(f"{feat}/(expose + 1)")
        col_percent_norm = col_percent + '_norm'
        df[col_percent_norm] = MinMaxScaler().fit_transform([[i] for i in df[feat].values]).flatten()
        df2[col_percent] = df2.eval(f"{feat}/(expose + 1)")
        q_score = df[col_percent].quantile(q)
        df2 = df2[df2[col_percent] <= q_score]
        # print(f'col:{col}, q_score:{q_score}, ')
    print(f"剩下占比：{len(df2)/len(df)}")

    feat = 'feedback'
    feat_percent = feat + '_percent'
    df[feat_percent] = df.eval(f'{feat}/(expose + 1)')
    df[feat_percent + '_norm'] = MinMaxScaler().fit_transform([[-i] for i in df[feat_percent].values]).flatten()
    df['post_score'] = df['click_percent_norm'] + df['comment_percent_norm'] + df['share_percent_norm'] + df['favor_percent_norm'] + df['feedback_percent_norm']

    prior_col = ['cover_pic_score', 'content_clarity', 'cc_ai_account_professional_score']
    for feat in prior_col:
        feat_norm = feat + '_norm'
        df[feat_norm] = MinMaxScaler().fit_transform([[i] for i in df[feat].values]).flatten()
        print(df[feat_norm].describe())

    df['prior_score'] = df[[feat+'_norm' for feat in prior_col]].sum(axis=1)

    df['alpha'] = df.expose.apply(cal_alpha)

    df['score'] = df.eval('post_score * alpha + (1-alpha) * prior_score')

    q10_score = df['score'].quantile(0.1)
    low_q_df = df[df['score'] < q10_score]
    print(len(low_q_df))

    # print(filter_low_q_df[['puin', 'score', 'alpha', 'post_score', 'prior_score']])

    puin_feature_columns = ['puin', 'expose', 'click_percent', 'comment_percent', 'share_percent', 'favor_percent', 'feedback_percent', 'follow_rate', 'follow', 'cover_pic_score', 'content_clarity', 'overall_level', 'cc_ai_account_professional_score', 'metrics_characteristic_account_tag', 'low_standard_75_num']

    low_puins_file = os.path.join(args.base_dir, args.sub_dir_name, "随机视频搬运号-0604-result.csv")
    low_puins_df = pd.read_csv(low_puins_file, header=0)
    columns = ['index', 'video_id', 'scene', 'puin', 'puin_name', 'overall_level',
       'metrics_7days_input', 'metrics_7days_st_kd', '启用率', 'accounttype',
       'puin_qb_7day_follow_rate', '转粉率(万）', 'follow_rate_rank',
       'floatExposure', 'feedsExposure', 'floatClick', 'feedsClick', 'ctr',
       'floatPlayTotalTime', 'floatVideoTotalTime', 'Unnamed: 20',
       'commentNum', 'shareNum', 'praiseNum', '点赞率', 'followSevenDay', '垂直度',
       '清晰度', '内容质量', 'manual_judge', '备注', 'Unnamed: 31', 'url']
    low_puins_df.columns = columns

    low_puins_df['puin'] = low_puins_df['puin'].astype(int)
    # puin_ids = low_puins_df['puin'].astype(int).astype(str).values
    # puin_ids = set(puin_ids)

    low_puin_columns = ['puin', 'manual_judge', '备注', 'Unnamed: 31']

    join_df = low_puins_df[low_puin_columns].merge(df[puin_feature_columns], how='inner', on='puin')
    # filter_low_q_df = join_df.sort_values(by='manual_judge').drop_duplicates(keep='first')
    # filter_low_q_df = join_df

    filter_low_q_df = filter(df)[puin_feature_columns].sample(1000)

    filter_low_q_df.rename(columns={
        'cc_ai_account_professional_score': '垂直度',
        'metrics_characteristic_account_tag': '搬运标记',
        'content_clarity': '清晰度',
        'manual_judge': '帐号质量评估(1差 2普通 3好)',
        'low_standard_75_num': '次低质标记数量'
    }).to_csv('low_quality_quin_v1.csv')

