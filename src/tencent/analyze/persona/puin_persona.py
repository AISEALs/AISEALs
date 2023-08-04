import os
import gc
import math
import argparse
import datetime
import multiprocessing as mp
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
    parser.add_argument('--scene', type=str, default='mini')
    parser.add_argument('--new_version', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--base_dir', type=str,
                        default='/Users/jiananliu/work/analyze/data',
                        help='please set')
    parser.add_argument('--sub_dir_name', type=str,
                        default='persona')
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


if __name__ == '__main__':
    # mp.set_start_method("forkserver")
    args = get_args()

    file_name = os.path.join(args.base_dir, args.sub_dir_name, "doc-puin-filterid.csv")
    df = pd.read_csv(file_name, sep=',')

    df['filter_ids'] = df.is_75_low_standard_mark_id.apply(lambda x: x.split("|"))

    # doc_filter_id = '10032' # 227_图文_75分_图文-娱乐类规则标题党-非图集打标
    doc_filter_id = '10029' # 122_图文_75分_图文标题党号
    mask = df.filter_ids.apply(lambda x: doc_filter_id in x)
    print(df[mask])

