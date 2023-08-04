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
    # scene = 'mini'  # 视频
    # scene = 'short'  # 视频
    # scene = 'feeds' # 图文
    parser.add_argument('--scene', type=str, default='feed')
    parser.add_argument('--new_version', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--base_dir', type=str,
                        default='/Users/jiananliu/Desktop/work/tencent/analyze/table',
                        help='please set')
    parser.add_argument('--sub_dir_name', type=str,
                        default='follow/follow_rate_dist')
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




if __name__ == '__main__':
    args = get_args()
    file_name = os.path.join(args.base_dir, args.sub_dir_name, "puin-follow-num.csv.gz")
    df = pd.read_csv(file_name, compression='gzip', sep='\t')

    high_follow_rate_thre = 0.0020
    df2 = df[df.eval(f'puin_30d_follow_rate >= {high_follow_rate_thre}')]
    print(f'30d_follow_rate > {high_follow_rate_thre}, puin 占比:{"%.2f" % (len(df2)/len(df))}, real expose 占比:{"%.2f" % (df2.real_expose.sum()/df.real_expose.sum())}')
    # print('gray_id: {} w>1 占比:{:.2%}'.format(gray_id, len(df2) / len(df1)))

    bins = [0, 5000, 10000, 100000, 500000, 1000000, 10000000000000]
    # bins = [round(i, 2) for i in np.arange(1, 5, 0.2)] + [5, 11]
    groups = pd.cut(df2.real_expose, bins)

    bb = df2.groupby(groups).size().reset_index()
    bb.columns = ['real_expose_group', 'puin_num']


    df2_total_expose = len(df2)
    bb['percent'] = bb.eval(f'puin_num/{df2_total_expose}')
    bb['percent'] = bb['percent'].apply(lambda x: format(x, '.2%'))
    bb.to_csv('result.csv', sep='\t')

    df3 = df2[df2.real_expose < 50000]
    # df3.sample(n=1000).to_csv('fr_w20_expose_5w.csv', sep='\t')
    df3.sort_values('real_expose').head(1000).to_csv('fr_w20_expose_5w.csv', sep='\t')





