import os
import gc
import math
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
    parser.add_argument('--scene', type=int, default=4)
    parser.add_argument('--new_version', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--base_dir', type=str,
                        default='/Users/jiananliu/Desktop/work/tencent/analyze/table',
                        help='please set')
    parser.add_argument('--sub_dir_name', type=str,
                        default='stg_id')
    parser.add_argument('--use_multi_proc', default=False, action='store_true',
                        help='is use multiprocessing lib to add speed')
    parser.add_argument('--multi_proc_num', type=int,
                        default=10)
    parser.add_argument('--show_table', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_pic', type=str,
                        default='expose_trend.png')
    parser.add_argument('--date', type=str,
                        default='20210704')
    parser.add_argument('--save_path', type=str,
                        default='result.csv')
    parser.add_argument('--task_type', type=int, required=False, default=3,
                        help="0: run middle data, gen result.csv.\n"
                             "1: run middle data, gen result.csv.\n"
                             "2: read result.csv to gen trend pic\n")

    return parser.parse_args()


def plot_stg_follow_dist():
    tong1 = 2390864
    tong2 = 2390867

    df2 = df1[['gray_id', 'stg_ids', 'follow_rate_7d']].explode(column='stg_ids')
    # aa = df2.groupby(['gray_id', 'stg_ids']).size().rename('num').reset_index()
    aa = df2.groupby(['gray_id', 'stg_ids'])['follow_rate_7d'].agg(['size', np.mean]).rename(columns={'size': 'num', 'mean': 'follow_rate'}).reset_index()
    bb = aa.pivot(index='stg_ids', columns='gray_id')
    bb = bb.fillna(0)
    bb.columns = bb.columns.to_flat_index().map(lambda x: '_'.join(map(str, x)))
    bb = bb.rename(columns={f'Y_{tong1}': 'v1', f'Y_{tong2}': 'v2'})

    bb2 = bb.reset_index()
    total_num_tong1 = bb2[f'num_{tong1}'].sum()
    total_num_tong2 = bb2[f'num_{tong2}'].sum()

    bb[f'percent_{tong1}'] = bb[f'num_{tong1}']/total_num_tong1
    bb[f'percent_{tong2}'] = bb[f'num_{tong2}']/total_num_tong2
    bb['percent_change'] = bb.eval(f'percent_{tong2} - percent_{tong1}')
    bb['num_change'] = bb.eval(f'num_{tong2} - num_{tong1}')
    bb['num_change_abs'] = bb['num_change'].apply(abs)
    bb['change_percent_abs'] = bb['num_change_abs']/(bb[f'num_{tong1}'] + 1.0)*1.0

    # bb2 = bb[bb.num_change_abs > 10000]
    # bb_head_10 = bb2.sort_values('change_percent_abs', ascending=False)[0: 10]
    bb2 = bb[bb.num_change_abs > 1000]
    bb_head_10 = bb2.sort_values('change_percent_abs', ascending=False)[0: 10]
    print(bb_head_10)
    bb_head_10 = bb_head_10.sort_values(f'num_{tong2}', ascending=False)
    bb_head_10[[f'percent_{tong1}', f'percent_{tong2}']].plot(kind='bar', title="exp comp by top change", figsize=(15, 10), legend=True, fontsize=12)
    plt.ylabel('output scale/100')
    plt.show()

    top_n = 2
    bb_head_n = bb.sort_values(f'percent_{tong2}', ascending=False)[0: 10*top_n]
    print(bb_head_n)
    print(f'top {top_n*10}策略号，曝光占比:{bb_head_n[f"num_{tong1}"].sum()/total_num_tong1*100}%')
    bb_head_n[[f'percent_{tong1}', f'percent_{tong2}']].plot(kind='bar', title="exp comp by top num", figsize=(10*top_n, 10), legend=True, fontsize=12)
    plt.ylabel('output scale/100')
    # bb_head_n['follow_rate_2290613'].plot(kind='line', secondary_y=True)

    plt.show()

def plot_stg_dist():
    tong1 = 2698066
    tong2 = 2698062
    # tong1 = 2290613
    # tong2 = 2290628
    df2 = df1[['gray_id', 'stg_ids']].explode(column='stg_ids')
    aa = df2.groupby(['gray_id', 'stg_ids']).size().rename('num').reset_index()
    aa['total_num'] = aa.groupby('gray_id')['num'].transform(sum)
    aa['Y'] = aa['num']/aa['total_num']*100

    bb = aa.pivot(index='stg_ids', columns='gray_id')
    bb = bb.fillna(0)
    bb.columns = bb.columns.to_flat_index().map(lambda x: '_'.join(map(str, x)))

    # bb = bb.reset_index()

    bb = bb.rename(columns={f'Y_{tong1}': 'base', f'Y_{tong2}': 'test'})
    # bb['Y_change'] = bb.eval('old - new')
    bb['num_change'] = bb.eval(f'num_{tong2} - num_{tong1}')
    bb['num_change_abs'] = bb['num_change'].apply(abs)
    bb['change_percent_abs'] = bb['num_change_abs']/(bb[f'num_{tong1}'] + 1.0)*1.0

    # bb2 = bb[bb.num_change_abs > 10000]
    # bb_head_10 = bb2.sort_values('change_percent_abs', ascending=False)[0: 10]
    # bb2 = bb[bb.num_change_abs > 1000]
    # bb_head_10 = bb2.sort_values('change_percent_abs', ascending=False)[0: 10]
    # print(bb_head_10)
    # bb_head_10 = bb_head_10.sort_values(f'num_{tong2}', ascending=False)
    # bb_head_10[[f'percent_{tong1}', f'percent_{tong2}']].plot(kind='bar', title="exp comp by top change", figsize=(15, 10), legend=True, fontsize=12)
    # plt.ylabel('output scale/100')
    # plt.show()

    top_n = 2
    bb_head_n = bb.sort_values('base', ascending=False)[0: 10*top_n]
    print(bb_head_n)
    # print(f'top {top_n*10}策略号，曝光占比:{bb_head_n[f"size_{tong1}"].sum()/total_num_tong1*100}%')
    bb_head_n[['base', 'test']].plot(kind='bar', title=f"exp comp by top{10*top_n} num", figsize=(10*top_n, 10), legend=True, fontsize=12)
    plt.ylabel('output scale/100')
    # bb_head_n['follow_rate_2290613'].plot(kind='line', secondary_y=True)

    plt.show()


if __name__ == '__main__':
    args = get_args()
    # file_name = os.path.join(args.base_dir, args.sub_dir_name, "stg_id_1day.csv")
    file_name = os.path.join(args.base_dir, args.sub_dir_name, "mini_stg_id_1hour_2021092512.csv")
    # total_df = pd.read_csv(file_name, compression='gzip', sep='\t')
    total_df = pd.read_csv(file_name, sep='\t', error_bad_lines=False)

    columns = [col.replace('t_sh_atta_v1_0bf00031859.', '') for col in total_df.columns]
    print(columns)
    total_df.columns = columns
    total_df['stg_ids'] = total_df['doc_reason_flag'].apply(lambda x: [] if x is np.nan else x.split(","))

    print('-' * 10 + 'server_name' + '-' * 10)
    # RECALL - 召回输出，RANK - 粗排输出，PREDICT - 精排输出，RERANK - 展控输出
    print(total_df.server_name.value_counts())
    server_name = 'RANK'
    df = total_df[total_df.server_name == server_name]
    print(f'filter by server_name: [{server_name}], {len(total_df)} => {len(df)}')

    print('-' * 10 + 'rec_scene' + '-' * 10)
    # 1-Feeds 2-视频秀 3-短视频浮层 4-小视频浮层 5-垂直频道 6-插入下一条 7-视频Portal 8-视频Portal下一条 9-影视综集锦 10-直播浮层
    # df1 = df[(df.rec_scene == 4) | (df.rec_scene == 3)]
    df1 = df[df.rec_scene == args.scene]
    print(f'filter by rec_scene: [short], {len(df)} => {len(df1)}')

    print('-' * 10 + 'gray_id' + '-' * 10)
    print(df1.gray_id.value_counts())

    plot_stg_dist()

    # doc_file_name = os.path.join(args.base_dir, args.sub_dir_name, "doc.csv")
    # doc_df = pd.read_csv(doc_file_name, sep='\t')
    # # doc_df['follow_rate_7d'] = doc_df.eval('(follow_7d + 0.1)/(expose_7d+2000)*10000')
    # # doc_df.columns = ['item_id', 'finish_rate', 'video_time']
    # join_df = df1.merge(doc_df, how='left', left_on='doc_id', right_on='item_id')
    # print('join doc poster feature.')
    # print(f'after join, follow_rate_7d filter num: {len(df1)} => {join_df.follow_rate_7d.count()}')
    # join_df = join_df.fillna(0)
    #
    # # 对比分位值变化
    # tong1 = 2290613
    # tong2 = 2290628
    # def f(tong):
    #     return join_df[join_df.eval(f'gray_id == {tong}')].follow_rate_7d.quantile([i for i in np.arange(0, 1, 0.1)])
    #
    # dd = pd.DataFrame({tong1: f(tong1), tong2: f(tong2)})
    # dd.plot(marker='o')
    # plt.xlabel('quantile')
    # plt.ylabel('followrate*1w')
    # plt.show()

    # quartiles = pd.cut(join_df[join_df.eval(f'gray_id == {tong2}')].follow_rate_7d, 10)
    #
    # # 定义聚合函数
    # def get_stats(group):
    #     return {'s': len(group)}
    #
    # # 分组统计
    # grouped = join_df[join_df.eval(f'gray_id == {tong1}')].follow_rate_7d.groupby(quartiles)
    # price_bucket_amount = grouped.apply(get_stats).unstack()
    # print(price_bucket_amount)
    #
    # np.arange(0, 1, 0.1)