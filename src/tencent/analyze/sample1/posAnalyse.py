import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

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
    parser.add_argument('--scene', type=str, default='feed')
    # RECALL - 召回输出，RANK - 粗排输出，PREDICT - 精排输出，RERANK - 展控输出
    parser.add_argument('--server_name', type=str, default='RERANK')
    parser.add_argument('--is_gray_exp', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--new_version', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--date', type=str, default='20211207x')
    parser.add_argument('--base_dir', type=str,
                        default='/Users/jiananliu/work/projects/data/analyze/table',
                        help='please set')
    parser.add_argument('--sub_dir_name', type=str,
                        default='sample100')
    parser.add_argument('--use_multi_proc', default=False, action='store_true',
                        help='is use multiprocessing lib to add speed')
    parser.add_argument('--multi_proc_num', type=int,
                        default=10)
    parser.add_argument('--show_table', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_pic', type=str,
                        default='expose_trend.png')
    parser.add_argument('--save_path', type=str,
                        default='result.csv')
    parser.add_argument('--task_type', type=int, required=False, default=3,
                        help="0: run middle data, gen result.csv.\n"
                             "1: run middle data, gen result.csv.\n"
                             "2: read result.csv to gen trend pic\n")

    return parser.parse_args()


def str2dict(str):
    m = dict()
    for x in str.split(';'):
        try:
            k = x.split(':')[0]
            if k in ['RhighFollowWeightScore', 'RdebugScore']:
                v = x.split(':')[1]
            # elif k == 'RdebugScore':
            #     v = x.split(':')[1]
            #     kv = dict(('S' + i.split('_')[0], float(i.split('_')[1])) for i in v.split('|'))
            #     m.update(kv)
            else:
                if '@1' in x or '@0' in x:
                    x = x.strip('@1')
                    x = x.strip('@0')
                v = float(x.split(':')[1])
            m[k] = v
        except:
            pass

    # m = dict(x.split(':') for x in str.split(';'))
    return m


def get_sample_df():
    if args.is_gray_exp:
       file_name = f"{args.server_name}-{args.date}-EXP.txt"
    else:
        file_name = f"{args.server_name}-{args.date}.txt"
    absolute_name = os.path.join(args.base_dir, args.sub_dir_name, file_name)
    # file_name = os.path.join(args.base_dir, args.sub_dir_name, f"{args.server_name}-2021111622.txt")
    total_df = pd.read_csv(absolute_name, sep='\t', error_bad_lines=False)

    columns = [col.replace('t_sh_atta_v1_0bf00031859.', '') for col in total_df.columns]
    print(f'read file:{file_name} success, line num:{len(total_df)}')
    print(f"columns: {columns}")
    total_df.columns = columns
    # total_df['stg_ids'] = total_df['doc_reason_flag'].apply(lambda x: [] if x is np.nan else x.split(","))
    return total_df


def process_rerank():
    print('短小视频数量：')
    print(total_df['busi'].value_counts())
    print('-' * 30)
    total_df['detail_doc_score_dict'] = total_df['detail_doc_score'].apply(str2dict)

    def parse_high_follow_weight_score(x):
        sp = x.split('_')
        if len(sp) == 3:
            w = float(sp[1])
            follow_rate = float(sp[2])
        else:
            w = 1.0 # default 1.0 on not adjust
            follow_rate = np.nan
        return w, follow_rate
    total_df['DhighFollowWeightScoreStr'] = total_df['detail_doc_score_dict'].map(lambda x: x.get('DhighFollowWeightScore', ''))
    total_df['follow_rate'] = total_df['detail_doc_score_dict'].map(lambda x: x.get('DitemFollowScore', ''))
    # total_df['follow_rate'] = total_df['DitemFollowScore']
    total_df['high_follow_weight'] = total_df['DhighFollowWeightScoreStr']
    if args.is_gray_exp:
        df2 = total_df.copy()
        df2['is_high_follow'] = df2['follow_rate'] > 0.0025
        agg_df = df2.groupby(['busi', 'gray_id'])['follow_rate'].agg(['mean', 'count'])
        agg_df['high_follow_count'] = df2.groupby(['busi', 'gray_id'])['is_high_follow'].agg('sum')
        agg_df = agg_df.reset_index()
        agg_df['total_exp'] = agg_df.groupby('gray_id')['count'].transform(np.sum)
        agg_df['high_follow_percent'] = agg_df['high_follow_count']/agg_df['count']
        agg_df['num_percent'] = agg_df['count']/agg_df['total_exp']
        agg_df['num_percent'] = agg_df['num_percent'].apply(lambda x: format(x, ".2%"))
        print(agg_df[['busi', 'gray_id', 'mean', 'num_percent', 'high_follow_percent']])

    print('粗排-高转粉加权分')
    total_num = len(total_df['high_follow_weight'])
    greater_df = total_df[total_df['high_follow_weight'] > 1.0]
    greater_num = len(greater_df)
    print(f'展控中-粗排加权占比：{"{:.2%}".format(greater_num/total_num)}, 均值:{"{:.2f}".format(greater_df.high_follow_weight.mean())}')
    print('-' * 30)

    cols = [col for col in total_df.loc[0]['detail_doc_score_dict'].keys() if col.startswith('D') or col.startswith('P')]
    # cols = [col for col in total_df.loc[0]['detail_doc_score_dict'].keys() if col.startswith('R')]
    for col in cols:
        if col in ['RhighFollowWeightScore']:
            continue
        total_df[col] = total_df['detail_doc_score_dict'].map(lambda x: x.get(col, np.nan)).replace([np.inf, -np.inf], np.nan)
        print('-' * 10 + col + '-' * 10)
        print(total_df[col].mean())

    # print(cols)
    # RhighFollowWeightScore：粗排-高转粉加权分
    detail_score_cols = {
        # 'RitemFollowScore': '粗排-资源转粉率',
        'RmergeModelScore': '粗排-融合分',
        'RfollowScore': '粗排-个性化关注分',
        'PmergeModelScore': '精排-融合分'
        # 'RmtMergeScore': '粗排-互动融合分'
        # 'RadjustWeightScore', 'RankIndex', 'RcasScore', 'RchdNumScore', 'RchdTimeScore', 'RctrScore',
        #  'RdiversityScore', 'RfeedbackScore', 'RfollowScore', 'RhighFollowWeightScore', 'RitemFollowScore',
        #  'RmergeModelScore', 'RpraiseScore', 'RpuinFollowScore', 'RratioScore', 'RreadCommentScore', 'RshareScore',
        #  'RskipScore', 'RtimeMergeScore', 'RtimeScore', 'RtransScore', 'RwriteCommentScore'
    }
    for col, info in detail_score_cols.items():
        print(info + ':' + col)
        print('mean:')
        print(total_df.groupby('busi')[col].mean())
        print('quantile:')
        print(total_df.groupby('busi')[col].describe())
        print('-' * 30)
    # print('展控-资源转粉率:')
    # print(total_df.groupby('busi')['DitemFollowScore'].describe())

    # total_df['DmergeModelScoreExceptFollow'] = total_df['DmergeModelScore']/total_df['RfollowScore']

    min_follow_rate_thresthold = 25
    greater_fr_percent = len(total_df[total_df['DitemFollowScore'] > min_follow_rate_thresthold/10000.0]) / len(total_df)
    print(f'fr >= w{min_follow_rate_thresthold} percent: {"{:.2%}".format(greater_fr_percent)}')

    filter_total_df = total_df[total_df.groupby('queryid')['DmergeModelScore'].transform('count') >= 95]
    filter_total_df['RANK'] = filter_total_df.groupby('queryid')['DmergeModelScore'].rank(method="first", ascending=False).astype(int)
    filter_total_df = filter_total_df[filter_total_df.RANK <= 100]

    for col in ['DmergeModelScore']: #, 'RmergeModelScoreExceptFollow', 'RfollowScore']:
        print('-'*10 + col + '-'*10)
        rank_df = filter_total_df.groupby('RANK')[col].agg(['mean', 'median'])
        df28 = rank_df.loc[[1] + [i for i in range(10, 110, 10)]]
        print(df28)
        # print(f'28位置score mean：{df28.loc[20]["mean"] / df28.loc[80]["mean"]}')
        # print(f'28位置score median：{df28.loc[20]["median"] / df28.loc[80]["median"]}')
        print(f'第20位置和第80位置的mean和median比值分别为：{"{:.2f}".format(df28.loc[20]["mean"] / df28.loc[80]["mean"])}和{"{:.2f}".format(df28.loc[20]["median"] / df28.loc[80]["median"])}')
        print(f'第1位置和第100位置的mean和median比值分别为：{"{:.2f}".format(df28.loc[1]["mean"] / df28.loc[100]["mean"])}和{"{:.2f}".format(df28.loc[1]["median"] / df28.loc[100]["median"])}')


def process_rank():
    print('短小视频数量：')
    print(total_df['busi'].value_counts())
    print('-' * 30)
    total_df['detail_doc_score_dict'] = total_df['detail_doc_score'].apply(str2dict)

    def parse_high_follow_weight_score(x):
        sp = x.split('_')
        if len(sp) == 3:
            w = float(sp[1])
            follow_rate = float(sp[2])
        else:
            w = 1.0 # default 1.0 on not adjust
            follow_rate = np.nan
        return w, follow_rate
    total_df['RhighFollowWeightScoreStr'] = total_df['detail_doc_score_dict'].map(lambda x: x.get('RhighFollowWeightScore', ''))
    total_df['follow_rate'] = total_df['RhighFollowWeightScoreStr'].apply(lambda x: parse_high_follow_weight_score(x)[1])
    total_df['is_high_follow'] = total_df['follow_rate'] > 0.0015
    agg_df = total_df[total_df.is_high_follow == True].groupby('busi')['follow_rate'].describe()

    total_df['high_follow_weight'] = total_df['RhighFollowWeightScoreStr'].apply(lambda x: parse_high_follow_weight_score(x)[0])
    print('粗排-高转粉加权分')
    total_num = len(total_df['high_follow_weight'])
    greater_df = total_df[total_df['high_follow_weight'] > 1.0]
    greater_num = len(greater_df)
    print(f'粗排中-粗排加权占比：{"{:.2%}".format(greater_num/total_num)}, 均值:{"{:.2f}".format(greater_df.high_follow_weight.mean())}')
    print('-' * 30)
    if args.is_gray_exp:
        print(f"粗排输出100条，AB实验转粉率对比：")
        print(total_df.groupby('gray_id')['follow_rate'].describe())
        print(f"粗排输出100条，AB实验高转粉调权值对比：")
        print(total_df.groupby('gray_id')['high_follow_weight'].describe())

    total_df['RdebugScore'] = total_df['detail_doc_score_dict'].map(lambda x: x.get('RdebugScore', ''))
    total_df['RdebugScoreDict'] = total_df['RdebugScore'].apply(lambda x: dict((i.split('_')[0], float(i.split('_')[1])) for i in x.split('|')))
    cols = ['RfinishScore', 'RtimeScore', 'RreTimeScore', 'RreSkipScore', 'RpeFinishScore', 'RpeSkipScore', 'RinteractMTScore', 'RrawScore', 'RfinalScore']
    total_times = 1.0
    for col in cols:
        # print('-'*10 + col + '-'*10)
        total_df['S' + col] = total_df['RdebugScoreDict'].apply(lambda x: x.get(col, 0.0))
        # print(total_df['S' + col].describe())
        time = total_df['S' + col].quantile(0.75)/total_df['S'+col].quantile(0.25)
        print(f'{col}: q75/q25={time}')
        if col != 'RfinalScore':
            total_times *= time
    print(f'total_times: {total_times}')

    # RhighFollowWeightScore：粗排-高转粉加权分
    detail_score_cols = {
        'RitemFollowScore': '粗排-资源转粉率',
        'RmergeModelScore': '粗排-融合分',
        # 'RmtMergeScore': '粗排-互动融合分'
        # 'RskipScore': '', #空的
        'RpraiseScore': '粗排-个性化点赞分',
        'RreadCommentScore': '粗排-个性化读评论分',
        'RshareScore': '粗排-个性化分享分',
        'RwriteCommentScore': '粗排-个性化写评论分',
        'RfollowScore': '粗排-个性化关注分',
    }
    for col, info in detail_score_cols.items():
        total_df[col] = total_df['detail_doc_score_dict'].map(lambda x: x.get(col, np.nan)).replace([np.inf, -np.inf],
                                                                                                    np.nan)
        print(info)
        print('mean:')
        print(total_df.groupby('busi')[col].mean())
        print('quantile:')
        print(total_df.groupby('busi')[col].describe())
        print('-' * 30)
    # print('展控-资源转粉率:')
    # print(total_df.groupby('busi')['DitemFollowScore'].describe())

    total_df['RmergeModelScoreExceptFollow'] = total_df['RmergeModelScore']/total_df['RfollowScore']

    min_follow_rate_thresthold = 10
    greater_fr_percent = len(total_df[total_df['RitemFollowScore'] > min_follow_rate_thresthold/10000.0]) / len(total_df)
    print(f'fr >= w{min_follow_rate_thresthold} percent: {"{:.2%}".format(greater_fr_percent)}')

    filter_total_df = total_df[total_df.groupby('queryid')['RmergeModelScore'].transform('count') >= 95]
    filter_total_df['RANK'] = filter_total_df.groupby('queryid')['RmergeModelScore'].rank(method="first", ascending=False).astype(int)
    filter_total_df = filter_total_df[filter_total_df.RANK <= 100]

    for col in ['RmergeModelScore', 'RmergeModelScoreExceptFollow', 'RfollowScore']:
        print('-'*10 + col + '-'*10)
        rank_df = filter_total_df.groupby('RANK')[col].agg(['mean', 'median'])
        df28 = rank_df.loc[[1] + [i for i in range(10, 110, 10)]]
        print(df28)
        # print(f'28位置score mean：{df28.loc[20]["mean"] / df28.loc[80]["mean"]}')
        # print(f'28位置score median：{df28.loc[20]["median"] / df28.loc[80]["median"]}')
        print(f'第20位置和第80位置的mean和median比值分别为：{"{:.2f}".format(df28.loc[20]["mean"] / df28.loc[80]["mean"])}和{"{:.2f}".format(df28.loc[20]["median"] / df28.loc[80]["median"])}')
        print(f'第1位置和第100位置的mean和median比值分别为：{"{:.2f}".format(df28.loc[1]["mean"] / df28.loc[100]["mean"])}和{"{:.2f}".format(df28.loc[1]["median"] / df28.loc[100]["median"])}')

    pdf = filter_total_df.pivot(index='queryid', columns='RANK', values='RmergeModelScore')
    pdf2 = (pdf[20]/pdf[80]).replace([np.inf, -np.inf], np.nan)


if __name__ == '__main__':
    args = get_args()
    total_df = get_sample_df()

    if args.server_name == 'RERANK':
        process_rerank()

    if args.server_name == 'RANK':
        process_rank()

    if args.server_name == 'RECALL':
        process_rerank()
        # total_df['detail_doc_score_dict'] = total_df['detail_doc_score'].apply(str2dict)
        # detail_score_cols = ['RadjustWeightScore', '', 'RmergeModelScore']
        # for col in detail_score_cols:
        #     total_df[col] = total_df['detail_doc_score_dict'].map(lambda x: x.get(col, np.nan)).replace(
        #         [np.inf, -np.inf], np.nan)
            # print(total_df[col].describe())
            # print(total_df.groupby('doc_showpos')[col].quantile(0.5))

