import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.tencent.analyze.analyze_table.big_publish_doc_analyze import read_data

# 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/table/content_account_dist'

# sep = '\t'
sep = '|'
def print_content_by_publish_num():
    df = read_data()

    # type = 3 # 1 图文 2 图集 3 短视频 19 小视频 21 搞笑
    # df = df[df.eval(f'type == {type}')]
    print(f'total account num:{df["puin"].count()}, ')
    print(df.info())

    # total_account_num = df[~df.metrics_7days_input.isna()]['puin'].count()
    # total_doc_num = df[~df.metrics_7days_input.isna()]['content_count'].sum()
    total_doc_num = df['video_id'].nunique()
    print(total_doc_num)
    total_expose_num = df['expose'].sum()

    # 0=0~10
    # 1=10~30
    # 2=30~50
    # 3=50~100
    # 4=100~500
    # 5=500+
    publish_levels = [0, 1, 100000]
    publish_levels_names = {'0_1': '[0, 10)', '1_100000': '[10, +)'}
    account_level = [1, 2, 3, 6]
    account_level_names = {'1_2': 1, '2_3': 2, '3_6': 4}

    for i in range(len(publish_levels) - 1):
        start = publish_levels[i]
        end = publish_levels[i + 1]
        for j in range(len(account_level) - 1):
            account_level_start = account_level[j]
            account_level_end = account_level[j + 1]
            df2 = df[df.eval(f'publish_level >= {start} and publish_level < {end} and account_level >= {account_level_start} and account_level < {account_level_end}')]
            publish_name = publish_levels_names[f"{start}_{end}"]
            account_name = account_level_names[f"{account_level_start}_{account_level_end}"]
            print(f'{sep}{account_name}{sep}{publish_name}{sep}{df2["puin"].nunique()}{sep}{df2["video_id"].nunique()}{sep}{"%.2f" % (df2["video_id"].nunique()/total_doc_num*100)}{sep}{"%.2f" % (df2["expose"].sum()/total_expose_num*100)}{sep}')

    print('-' * 20)
    publish_levels = [0, 1, 3, 4, 5, 100000]
    # publish_levels = [0, 10, 50, 100, 500, 1000, 100000]
    publish_levels_names = {'0_1': '[0, 10)', '1_3': '[10, 50)', '3_4': '[50, 100)', '4_5': '[100, 500)', '5_100000': '[500, +)'}
    account_level = [3, 6]
    account_level_names = {'1_2': 1, '2_3': 2, '3_6': 4}

    for i in range(len(publish_levels) - 1):
        start = publish_levels[i]
        end = publish_levels[i + 1]
        for j in range(len(account_level) - 1):
            account_level_start = account_level[j]
            account_level_end = account_level[j + 1]
            df2 = df[df.eval(f'publish_level >= {start} and publish_level < {end} and account_level >= {account_level_start} and account_level < {account_level_end}')]
            publish_name = publish_levels_names[f"{start}_{end}"]
            account_name = account_level_names[f"{account_level_start}_{account_level_end}"]
            print(f'{sep}{account_name}{sep}{publish_name}{sep}{df2["puin"].nunique()}{sep}{df2["video_id"].nunique()}{sep}{"%.2f" % (df2["video_id"].nunique()/total_doc_num*100)}{sep}{"%.2f" % (df2["expose"].sum()/total_expose_num*100)}{sep}')
    # for i in range(len(publish_levels) - 1):
    #     start = publish_levels[i]
    #     end = publish_levels[i + 1]
    #     for overall_level in [3, 6]:
    #         df2 = df[df.eval(f'publish_level >= {start} and publish_level < {end} and account_level == {overall_level}')]
    #         print(f'{sep}{overall_level}{sep}[{start}, {end}){sep}{df2["puin"].nunique()}{sep}{df2["video_id"].nunique()}{sep}{"%.2f" % (df2["video_id"].nunique()/total_doc_num*100)}{sep}{"%.2f" % (df2["expose"].sum()/total_expose_num*100)}{sep}')


def print_content_by_follow_rate():
    df = read_data()

    # type = 19# 1 图文 2 图集 3 短视频 19 小视频 21 搞笑
    # df = df[df.eval(f'type == {type}')]
    # print(f'total account num:{df["puin"].count()}, ')
    # print(df.info())

    df.eval('followrate = fol_fans_num_daily/(con_real_clk+1)*10000', inplace=True)
    # print(f'total account num:{df["puin"].count()}, ')

    # total_account_num = df[~df.metrics_7days_input.isna()]['puin'].count()
    total_doc_num = df['content_count'].sum()
    print(total_doc_num)
    total_expose_num = df['click'].sum()

    levels = [0, 1, 3, 5, 10, 100000]
    overall_levels = [1, 2 , 4]
    for i in range(len(levels) - 1):
        start = levels[i]
        end = levels[i + 1]
        for overall_level in overall_levels:
            df2 = df[df.eval(f'followrate >= {start} and followrate < {end} and overall_level == {overall_level}')]
        # print(f'{start}~{end}:', df2['puin'].count(), df2['content_count'].sum())
            print(f'{sep}{overall_level}{sep}[{start}, {end}){sep}{df2["puin"].count()}{sep}{df2["content_count"].sum()}{sep}{"%.2f" % (df2["content_count"].sum()/total_doc_num*100)}{sep}{"%.2f" % (df2["click"].sum()/total_expose_num*100)}{sep}')

    print('-' * 20)
    levels = [0, 1, 3, 5, 10, 100000]
    for i in range(len(levels) - 1):
        start = levels[i]
        end = levels[i + 1]
        for overall_level in [4]:
            df2 = df[df.eval(f'followrate >= {start} and followrate < {end} and overall_level == {overall_level}')]
            # print(f'{start}~{end}:', df2['puin'].count(), df2['content_count'].sum())
            print(
                f'{sep}{overall_level}{sep}[{start}, {end}){sep}{df2["puin"].count()}{sep}{df2["content_count"].sum()}{sep}{"%.2f" % (df2["content_count"].sum() / total_doc_num * 100)}{sep}{"%.2f" % (df2["click"].sum()/total_expose_num*100)}{sep}')


if __name__ == '__main__':

    print_content_by_publish_num()
    # print_content_by_follow_rate()