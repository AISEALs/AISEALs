import os
import math
import argparse
import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='BigPublishDoc-Profile-v2')
    # scene = 'mini'  # 视频
    # scene = 'short'  # 视频
    # scene = 'feeds' # 图文
    parser.add_argument('--scene', type=str, default='mini')
    parser.add_argument('--base_dir', type=str,
                        default='/Users/jiananliu/Desktop/work/tencent/analyze/table',
                        help='please set')
    parser.add_argument('--sub_dir_name', type=str,
                        default='big_publish_doc_real')
    parser.add_argument('--use_multi_proc', default=False, action='store_true',
                        help='is use multiprocessing lib to add speed')
    parser.add_argument('--multi_proc_num', type=int,
                        default=10)
    parser.add_argument('--show_table', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_pic', type=str,
                        default='expose_trend.png')
    parser.add_argument('--save_path', type=str,
                        default='result.csv')
    # parser.add_argument('--task_type', type=int, required=True,
    #                     help="0: run middle data, gen result.csv.\n"
    #                          "1: run middle data, gen result.csv.\n"
    #                          "2: read result.csv to gen trend pic\n")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # expose_type = '后台曝光'
    expose_type = '真实曝光'
    file_name = os.path.join(args.base_dir, args.sub_dir_name, f'{args.scene}-{expose_type}.csv')
    # file_name = os.path.join(args.base_dir, args.sub_dir_name, '真实曝光.csv')

    df = pd.read_csv(file_name, sep='\t')

    not_null_df = df.dropna()

    total_expose = not_null_df['c'].sum()
    big_expose = not_null_df[not_null_df.publish_level >= 1]['c'].sum()
    print(f'{args.scene} 大发文[10, ～）{expose_type} 占比： {"%.2f" % (big_expose/total_expose*100)}%')
