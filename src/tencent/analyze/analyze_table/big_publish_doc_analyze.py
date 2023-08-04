import os
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
                        default='/Users/jiananliu/Desktop/work/tencent/analyze/table',
                        help='please set')
    parser.add_argument('--sub_dir_name', type=str,
                        default='big_publish_percent')
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


def effect_area_DLevel(raw_df):
    # file_name = os.path.join(base_dir, sub_dir_name, "item.csv")
    # raw_df = pd.read_csv(file_name, sep='\t')

    raw_df = raw_df[~raw_df.puin_id.isna()]
    # raw_df['puin_id'].astype(dtype=str)
    raw_df['puin'] = raw_df['puin_id'].apply(lambda x: str(int(x)))
    # raw_df = raw_df[['puin', 'account_level', 'video_count', 'expose', 'overall_level']]
    print(raw_df.agg(['nunique', 'count']))

    # -- metrics_7days_input:7天发文量
    # -- metrics_7days_st_kd:7天启用量
    # -- qb_7day_total_exp_cnt:曝光量
    # metrics_day_st_kd  启动量
    # metrics_day_input 发文量
    # qb_day_total_exp_cnt  曝光量

    df = raw_df

    not_null_df = df.dropna()
    print_result(not_null_df, df, 'not null')

    account_level = 1  # D级账生效
    df = df[df.eval(f'account_level == {account_level}')]

    print(f'D级账号:')
    print(
        f'大盘账号数量:{raw_df.puin.nunique()} => {df.puin.nunique()}, 占比:{"%.4f" % (df.puin.nunique() / raw_df.puin.nunique() * 100)}%')
    tmp_df = df[['puin', 'expose']].groupby('puin').sum()
    print(f'实际符合条件D级账号数量: {tmp_df[tmp_df.eval("expose>5000")].count()}')

    print(
        f'影响资源数量：{raw_df.video_id.nunique()} => {df.video_id.nunique()}, 占比:{"%.4f" % (df.video_id.nunique() / raw_df.video_id.nunique() * 100)}%')

    print(
        f'expose数量：{raw_df["expose"].sum()} => {df["expose"].sum()}, 占比:{"%.5f" % (df["expose"].sum() / raw_df["expose"].sum() * 100)}%')


def parse_start_use(x):
    try:
        if isinstance(x, float) and np.isnan(x):
            return x
        elif isinstance(x, str) and x is not None:
            return (str(x).split('#')[1]).split('%')[0]
        else:
            print('error start_use type neight float nor str: ' + x)
            return None
    except:
        print('error start_use:' + x)
        return None


def trans_puin_str(x):
    try:
        if isinstance(x, float):
            if np.isnan(x):
                return ''
            else:
                return str(int(x))
        elif isinstance(x, str) and x is not None:
            return str(int(x))
        else:
            print('error puin neight float nor str: ' + x)
            return ''
    except:
        print('error puin:' + str(x))
        return ''


def read_data(args, date):
    if args.new_version:
        file_name = os.path.join(args.base_dir, args.sub_dir_name,
                                 f"profile.t_sd_profile_mvideo_account_expose_new.{date}.gz")
    else:
        file_name = os.path.join(args.base_dir, args.sub_dir_name, f"mvideo.t_sd_mvideo_account_expose_tmp.{date}.gz")
    df = pd.read_csv(file_name, compression='gzip', sep='|')
    if args.new_version:
        df.columns = ['scene', 'puin_id', 'video_id', 'expose', 'account_level', 'publish_level',
                      'metrics_characteristic_account_tag', 'account_type']
    else:
        if len(df.columns) > 7:
            df.columns = ['puin_id', 'account_level', 'start_use', 'publish_level', 'video_id', 'expose', 'scene',
                          'is_banyun', 'is_link']
        else:
            df.columns = ['puin_id', 'account_level', 'start_use', 'publish_level', 'video_id', 'expose', 'scene']

    df['puin'] = df['puin_id'].apply(lambda x: trans_puin_str(x))
    # df['start_use'] = df['start_use'].apply(lambda x: parse_start_use(x))
    # df['start_use'] = pd.to_numeric(df['start_use'], errors='coerce').astype(np.float)
    df['account_level'] = pd.to_numeric(df['account_level'], errors='coerce').astype(np.float)
    df['account_level'].fillna(-1, inplace=True)
    if args.new_version:
        df['metrics_characteristic_account_tag'] = pd.to_numeric(df['metrics_characteristic_account_tag'],
                                                                 errors='coerce').astype(np.float)
        df['account_type'] = pd.to_numeric(df['account_type'], errors='coerce').astype(np.float)
        df['account_type'].fillna(-1, inplace=True)

    print(f'read date:{date} success')
    print(df.count())
    df = df[df.eval(f'scene == "{args.scene}"')]
    print(df.count())
    return df


def effect_area(df):
    for col in ['puin', 'account_level', 'publish_level', 'video_id', 'expose']:
        null_df = df[(df[col].isnull()) | (df[col] == '\\N') | (df[col].isna()) | (df[col] == '') | (df[col] == -1) | (
                    df[col] == 0)]
        # not_null_df = df[~df[col].isnull()]
        print(f'{col} is null, expose占比：{null_df["expose"].sum() / df["expose"].sum() * 100}%')
        print(f'{col} is null, count占比：{null_df["puin"].count() / df["puin"].count() * 100}%')

    print('-' * 10 + 'not_null' + '-' * 10)
    not_null_df = df.dropna()
    print_result(not_null_df, df, 'not null')

    print('-' * 20)
    # print(df.groupby('account_level').agg({'expose': 'sum'})['expose'])
    for account_level in [-1, 0, 1, 2, 3, 4, 5]:
        tmp_df = df[df.eval(f'account_level == {account_level}')]
        if len(tmp_df) == 0:
            continue
        print_result(tmp_df, df, f'account_level={account_level}')

    print('-'*10 + 'for print ACD percent' + '-'*10)
    acd_df = not_null_df[not_null_df.account_level.isin([1, 2, 4])]
    acd_df.drop_duplicates(subset=['puin_id'], keep='last', inplace=True)
    for account_level in [1, 2, 4]:
        tmp_df = acd_df[acd_df.eval(f'account_level == {account_level}')]
        if len(tmp_df) == 0:
            continue
        print_result(tmp_df, acd_df, f'account_level={account_level}')

    # S:5,A:4,B:3,C:4,D:5
    # CD级账号
    cd_df = not_null_df[not_null_df.eval(f'account_level == 1 or account_level == 2')]
    a_df = not_null_df[not_null_df.eval(f'account_level == 4')]

    publish_level2range = {
        0: "[0, 10)",
        1: "[10, 30)",
        2: "[30, 50)",
        3: "[50, 100)",
        4: "[100, 500)",
        5: "[500, ~)",
    }
    print('-' * 10 + 'publish_level' + '-' * 10)
    records = []
    for publish_level, min2max in publish_level2range.items():
        df2 = not_null_df[not_null_df.eval(f'publish_level == {publish_level}')]
        result = print_result(df2, not_null_df, f'all && {min2max}')
        result['publish_level'] = publish_level
        result['account_level'] = 'all'
        records.append(result)

    for publish_level, min2max in publish_level2range.items():
        df2 = a_df[a_df.eval(f'publish_level == {publish_level}')]
        result = print_result(df2, not_null_df, f'A && {min2max}')
        result['publish_level'] = publish_level
        result['account_level'] = 'A'
        records.append(result)

    for publish_level, min2max in publish_level2range.items():
        df2 = cd_df[cd_df.eval(f'publish_level == {publish_level}')]
        result = print_result(df2, not_null_df, f'CD && {min2max}')
        result['publish_level'] = publish_level
        result['account_level'] = 'CD'
        records.append(result)

    for publish_level, min2max in [(1, "[10, ~)")]:
        df2 = a_df[a_df.eval(f'publish_level >= {publish_level}')]
        print_result(df2, not_null_df, f'A && {min2max}')

    for publish_level, min2max in [(1, "[10, ~)")]:
        df2 = cd_df[cd_df.eval(f'publish_level >= {publish_level}')]
        print_result(df2, not_null_df, f'CD && {min2max}')

    if args.new_version and False:
        print('-------account_type---------')
        for publish_level, min2max in publish_level2range.items():
            df_publv = a_df[a_df.eval(f'publish_level == {publish_level}')]
            for account_type in [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                df_publv_acctype = df_publv[df_publv.eval(f'account_type == {account_type}')]
                print_result(df_publv_acctype, not_null_df, f'A && {min2max} && account_type={account_type}')
            print(df_publv.expose.sum() / not_null_df.expose.sum() * 100)
        print(a_df.expose.sum() / not_null_df.expose.sum() * 100)

        print('-------account_type---------')


    return pd.DataFrame.from_records(records)


def effect_some_area(df):
    not_null_df = df.dropna()
    puins2features = {
        'puins1': {
            'account_level': '3|4|5',
            'publish_level': 2,
            # 'account_type' : '0|1|5',
            'metrics_characteristic_account_tag': '0|1'
        },
        'puins2': {
            'account_level': '3|4|5',
            'publish_level': 1,
            # 'account_type': '0|1|5',
            'metrics_characteristic_account_tag': '0|1'
        }
    }

    def expose_func(df):
        return df.expose.sum()

    for puins, features in puins2features.items():
        df = not_null_df
        print('=' * 10 + puins + ':' + str(features) + '=' * 10)

        total_expose = expose_func(df)
        print(f'total: {total_expose}', end=' ')
        # 0=0~10
        # 1=10~30
        # 2=30~50
        # 3=50~100
        # 4=100~500
        # 5=500+
        col = 'publish_level'
        if col in features:
            publish_doc_level = features[col]
            df = df[df.eval(f'{col} >= {publish_doc_level}')]  # 发文量
            last_expose = expose_func(df)
            print(f'-> {col}: {last_expose, "%.2f" % (last_expose / total_expose * 100)}%', end=' ')

        col = 'account_level'
        if col in features:
            account_level = features[col]
            if isinstance(account_level, int):
                df = df[df.eval(f'{col} < {account_level} and {col} > 0')]
            else:
                account_levels = account_level.split('|')
                df = df[df.account_level.astype('int').astype('str').isin(account_levels)]
            last_expose = expose_func(df)
            print(f'-> {col}: {last_expose, "%.2f" % (last_expose / total_expose * 100)}%', end=' ')

        col = 'account_type'
        if col in features:
            account_type = features[col]
            account_types = account_type.split('|')
            df = df[df.account_type.astype('int').astype('str').isin(account_types)]
            last_expose = expose_func(df)
            print(f'-> {col}: {last_expose, "%.2f" % (last_expose / total_expose * 100)}%', end='\n')

        col = 'metrics_characteristic_account_tag'
        if col in features:
            account_type = features[col]
            metrics_characteristic_account_tags = account_type.split('|')
            df = df[df[col].astype('int').astype('str').isin(metrics_characteristic_account_tags)]
            last_expose = expose_func(df)
            print(f'-> {col}: {last_expose, "%.2f" % (last_expose / total_expose * 100)}%', end='\n')
        print(f'过滤数量变化：{not_null_df.expose.sum()} -> {df.expose.sum()}')
        print_result(df, not_null_df, puins)



def tong_compare_expose(args, date):
    file_name = os.path.join(args.base_dir, args.sub_dir_name,
                             f"mvideo.t_sd_mvideo_account_item_tong_expose_tmp.{date}.gz")
    raw_df = pd.read_csv(file_name, compression='gzip', sep='|')
    if len(raw_df.columns) <= 7:
        raw_df.columns = ['puin_id', 'account_level', 'start_use', 'publish_level', 'tong', 'expose', 'scene']
    elif len(raw_df.columns) == 9:
        raw_df.columns = ['puin_id', 'account_level', 'start_use', 'publish_level', 'tong', 'expose', 'scene',
                          'is_banyun', 'is_link']
    else:
        raw_df.columns = ['scene', 'tong', 'puin_id', 'expose', 'account_level', 'publish_level', 'metrics_characteristic_account_tag', 'account_type']

    raw_df = raw_df[raw_df.eval(f'scene == "{args.scene}"')]

    raw_df['puin'] = raw_df['puin_id'].apply(lambda x: str(int(x)) if not math.isnan(x) else "")
    # raw_df['start_use'] = raw_df['start_use'].apply(lambda x: parse_start_use(x))
    # raw_df['start_use'] = pd.to_numeric(raw_df['start_use'], errors='coerce').astype(np.float)
    raw_df['account_level'].fillna(-1, inplace=True)
    raw_df['account_level'] = pd.to_numeric(raw_df['account_level'], errors='coerce').astype(np.float)
    raw_df = raw_df.dropna()

    tong2features = {
        "short": {
            '1376164': {
                'account_level': '2|1',
                'publish_level': 1,
            },
            '1619177': {
                'account_level': '3|4|5',
                'publish_level': 4,
            },
            '1619172': {
                'account_level': '3|4|5',
                'publish_level': 4,
                'account_type': '0|1|5'
            },
            '1619164': {
                'account_level': '3|4|5',
                'publish_level': 3
            },
            '1453609': {
                'account_level': '3|4|5',
                'publish_level': 3,
                'account_type': '0|1|5'
            },
            '1750022': {
                'account_level': '3|4|5',
                'publish_level': 2,
            },
            '1750017': {
                'account_level': '3|4|5',
                'publish_level': 2,
                'account_type': '0|1|5'
            },
            '1750015': {
                'account_level': '3|4|5',
                'publish_level': 1,
            },
            '1750013': {
                'account_level': '3|4|5',
                'publish_level': 1,
                'account_type': '0|1|5'
            },
        },
        "mini": {
            '1376170': {
                'account_level': '2|1',
                'publish_level': 3,
            },
            '1750009': {
                'account_level': '2|1',
                'publish_level': 2,
            },
            '1750007': {
                'account_level': '2|1',
                'publish_level': 1,
            },
            '1750001': {
                'publish_level': 1,
                'account_level': '3|4|5',
                'account_type': '0|1|5'
            },
            '1749999': {
                'account_level': '3|4|5',
                'publish_level': 2,
                'account_type': '0|1|5'
            },
        }
    }

    def expose_func(df):
        return df[df.eval(f'tong == "{tong1}"')].expose.sum()

    print('实验桶效果：')
    for tong1, features in tong2features[args.scene].items():
        print('=' * 10 + tong1 + ':' + str(features) + '=' * 10)

        tong2 = str(int(tong1) - 1)

        df = raw_df
        total_expose = expose_func(df)
        print(f'total: {total_expose}', end=' ')
        tong_raw_df = df.groupby('tong')['expose'].sum()
        # 0=0~10
        # 1=10~30
        # 2=30~50
        # 3=50~100
        # 4=100~500
        # 5=500+
        col = 'publish_level'
        if col in features:
            publish_doc_level = features[col]
            df = df[df.eval(f'{col} >= {publish_doc_level}')]  # 发文量
            last_expose = expose_func(df)
            print(f'-> {col}: {last_expose, "%.2f" % (last_expose/total_expose*100)}%', end=' ')

        col = 'account_level'
        if col in features:
            account_level = features[col]
            if isinstance(account_level, int):
                df = df[df.eval(f'{col} < {account_level} and {col} > 0')]
            else:
                account_levels = account_level.split('|')
                df = df[df.account_level.astype('int').astype('str').isin(account_levels)]
            last_expose = expose_func(df)
            print(f'-> {col}: {last_expose, "%.2f" % (last_expose/total_expose*100)}%', end=' ')

        col = 'account_type'
        if col in features:
            account_type = features[col]
            account_types = account_type.split('|')
            df = df[df.account_type.astype('int').astype('str').isin(account_types)]
            last_expose = expose_func(df)
            print(f'-> {col}: {last_expose, "%.2f" % (last_expose/total_expose*100)}%', end='\n')

        print(f'过滤数量变化：{raw_df.expose.sum()} -> {df.expose.sum()}')
        df_tong = df.groupby(['tong']).agg({"expose": "sum"})
        df_expose = df_tong['expose']
        if tong1 not in df_expose:
            print(f'{tong1} is not in, please add')
            continue
        expose1 = df_expose[tong1]
        expose2 = df_expose[tong2]
        print(
            f'tong:{tong1}, 实验桶后台曝光:{expose1}({"%.3f" % (expose1 / tong_raw_df[tong1] * 100)}%), 对照桶后台曝光：{expose2}({"%.3f" % (expose2 / tong_raw_df[tong2] * 100)}%), 降低:{"%.4f" % ((expose2 - expose1) / expose2)}')

        tong1_df = df[df.eval(f'tong == "{tong1}"')]
        tong2_df = df[df.eval(f'tong == "{tong2}"')]
        print(tong1_df.groupby('publish_level').agg({'expose': 'sum'}))
        print(tong2_df.groupby('publish_level').agg({'expose': 'sum'}))

        print('过滤后账号:')
        print_tong_result(df[df.eval(f'tong == "{tong1}"')], raw_df[raw_df.eval(f'tong == "{tong1}"')],
                          f'tong: {tong1}')
        print_tong_result(df[df.eval(f'tong == "{tong2}"')], raw_df[raw_df.eval(f'tong == "{tong2}"')],
                          f'tong: {tong2}')

        print('-' * 20)
        tong_df = raw_df.groupby(['tong']).agg({'expose': 'sum'})['expose']
        tong1_total_expose = tong_df[tong1]
        tong2_total_expose = tong_df[tong2]

        level_dist_df = raw_df.groupby(['tong', 'account_level']).agg({'expose': 'sum'})['expose']
        # print(level_dist_df[tong1])
        # print(tong1_total_expose)
        # print(level_dist_df[tong2])
        # print(tong2_total_expose)
        tong1_df = level_dist_df[tong1].apply(lambda x: x / tong1_total_expose)
        tong2_df = level_dist_df[tong2].apply(lambda x: x / tong2_total_expose)
        print(tong1_df)
        print(tong2_df)


def stats(df):
    print(df.puin.nunique())
    d_1w_df = df[df.eval('account_level == 1 and expose > 10000')]
    print(d_1w_df.puin.nunique())


def print_result(df1, df, desc):
    account_num = df1.puin_id.nunique()
    account_num_all = df.puin_id.nunique()
    account_percent = "%.2f" % (account_num / account_num_all * 100)

    item_num = df1.video_id.nunique()
    item_num_all = df.video_id.nunique()
    item_percent = "%.2f" % (item_num / item_num_all * 100)

    expose_num = df1.expose.sum()
    expose_num_all = df.expose.sum()
    expose_percent = "%.2f" % (expose_num / expose_num_all * 100)
    if not args.show_table:
        print(f'{desc}: \n'
              f'账号:{account_num}, 大盘:{account_num_all}, '
              f'账号占比:{account_percent}%, '
              f'资源:{item_num}, 大盘:{item_num_all}, '
              f'资源占比:{item_percent}%, '
              f'曝光:{expose_num}, 大盘曝光:{expose_num_all}, '
              f'曝光占比:{expose_percent}%')
    else:
        print(f'|{desc}'
              f'|{account_num}/{account_num_all}'
              f'|{account_percent}%'
              f'|{item_num}/{item_num_all}'
              f'|{item_percent}%'
              f'|{expose_num}/{expose_num_all}'
              f'|{expose_percent}%|')

    return {'account_num': account_num,
            'account_num_all': account_num_all,
            'account_percent': account_percent,
            'item_num': item_num,
            'item_num_all': item_num_all,
            'item_percent': item_percent,
            'expose_num': expose_num,
            'expose_num_all': expose_num_all,
            'expose_percent': expose_percent}


def print_tong_result(df1, df, desc):
    print(f'{desc}: \n'
          f'账号:{df1.puin_id.nunique()}, 大盘:{df.puin_id.nunique()}, '
          f'账号占比:{"%.4f" % (df1.puin_id.nunique() / df.puin_id.nunique() * 100)}%, '
          # f'资源:{df1.video_id.nunique()}, 大盘:{df.video_id.nunique()}, '
          # f'资源占比:{"%.4f" % (df1.video_id.nunique() / df.video_id.nunique() * 100)}%, '
          f'曝光:{df1.expose.sum()}, 大盘曝光:{df.expose.sum()}, '
          f'曝光占比:{"%.4f" % (df1.expose.sum() / df.expose.sum() * 100)}%')


def get_result(args, dateStr):
    process_name = mp.current_process().name
    print("Current process start:", process_name, ", Input dateStr:", dateStr)
    df = read_data(args, dateStr)
    df2 = effect_area(df)
    df2['date'] = dateStr
    print("Process end:", process_name, ", Input dateStr:", dateStr)
    return df2


def read_middle_data(args):
    idx_list = args.date_list
    if isinstance(idx_list, list):
        if args.use_multi_proc:
            with mp.Pool(args.multi_proc_num) as pool:
                results = [pool.apply_async(func=get_result, args=(args, str(date),)) for date in idx_list]
                df_list = [p.get() for p in results]
                # pool.join()
                print(f'debug, df_list len:{len(df_list)}')
                df = pd.concat(df_list, axis=0, ignore_index=True)
        else:
            df_list = [get_result(args, dateStr=str(date)) for date in idx_list]
            df = pd.concat(df_list, axis=0, ignore_index=True)
    else:
        df = read_data(args, idx_list)

    return df


def expose_trend_figure(account_level, publish_level_list, y_axis_col='expose_percent'):
    if not isinstance(publish_level_list, list):
        publish_level_list = [publish_level_list]
    print('-' * 20)
    df = pd.read_csv(args.save_path)
    df.columns = ['id', 'account_num', 'account_num_all', 'account_percent',
                  'item_num', 'item_num_all', 'item_percent', 'expose_num',
                  'expose_num_all', 'expose_percent', 'publish_level', 'account_level',
                  'date']

    publish_level2range = {
        0: "0",
        1: "10",
        2: "30",
        3: "50",
        4: "100",
        5: "500",
        6: "~"
    }

    publish_start = publish_level2range[publish_level_list[0]]
    publish_end = publish_level2range[publish_level_list[-1] + 1]
    title = f'{account_level}, publish_level:[{publish_start}, {publish_end}) exposure ratio'
    df = df[(df.publish_level.isin(publish_level_list)) & (df.account_level == account_level) & (
        df.date.isin(args.date_list))]
    df['date'] = df['date'].apply(lambda x: f"{str(x)[-4:-2]}-{str(x)[-2:]}")
    df[y_axis_col] = df[y_axis_col].apply(pd.to_numeric)

    filter_df = df[['date', y_axis_col]].sort_values(by='date')
    date_df = filter_df.groupby('date').sum().reset_index()
    plt.plot(date_df['date'], date_df[y_axis_col], marker='o')
    # plt.figure(figsize=(, 4))
    plt.xticks(rotation=-90)
    plt.xlabel(f'date')
    plt.ylabel('expose ratio')
    plt.title(f'{title}')
    # plt.legend()
    plt.savefig(args.save_pic)
    plt.show()


def account_evaluate():
    file_name = os.path.join(args.base_dir, args.sub_dir_name, "puin_100.csv")
    account_df = pd.read_csv(file_name, sep='\t', dtype={'puin': 'str'})

    task_type = 1
    if task_type == 1:
        date = 20210307
        df = read_data(args=args, date=date)
        cd_df = df[df.eval(f'account_level == 1 or account_level == 2')]
        del df
        import gc
        gc.collect()
        cd_df_100 = cd_df[cd_df.publish_level >= 4]
        puin_statics_features_df = cd_df_100.groupby('puin').agg({'video_id': 'nunique', 'expose': 'sum'}).rename(
            columns={'video_id': 'video_num', 'expose': 'total_expose'}).reset_index()
        final_df = puin_statics_features_df.merge(account_df, on=['puin'], how='inner')

        df2 = account_df.set_index('puin').join(puin_statics_features_df.set_index('puin'), how='inner').reset_index()
        # final_df[['puin', 'puin_name', 'metrics_day_input', 'metrics_7days_input', 'accounttype', 'video_num',
        #           'total_expose']].to_csv(save_file_name)
        print(final_df)
    else:
        save_file_name = os.path.join(args.base_dir, args.sub_dir_name, '小视频大发文测评.csv')
        df = pd.read_csv(save_file_name)
        df.groupby('accounttype').agg({'puin': 'count', 'video_num': 'sum', 'total_expose': 'sum'})
        df2 = df['抽选评价']
        for label in ['无聊', '杂乱', '搬运']:
            df3 = df2[df2.str.contains(label)]
            prob = df3.count() / df2.count()
            print(f'{label}: {df3.count()} {prob}')

        df['ok_prob'] = df['人审通过率'].apply(lambda x: float(str(x).replace("%", "")))
        for min, max in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
            sub_df = df[df.eval(f"ok_prob >= {min} and ok_prob < {max}")]
            prob = sub_df['ok_prob'].count() / df['ok_prob'].count()
            print(f'[{min}, {max}): {sub_df["ok_prob"].count()} {"%.1f" % (prob * 100)}%')


def date_range(start_date, end_date):
    for i in range((end_date - start_date).days + 1):
        day = start_date + datetime.timedelta(days=i)
        yield day.strftime("%Y%m%d")


def run_join_expose():
    raw_df = pd.read_csv('run_join_expose.csv', sep=',')
    raw_df.columns = ['scene_tmp', 'account_level', 'publish_level', 'accounttype', 'pv']
    raw_df['account_level'].fillna(-1, inplace=True)
    raw_df['account_level'] = pd.to_numeric(raw_df['account_level'], errors='coerce').astype(np.float)
    raw_df['accounttype'] = pd.to_numeric(raw_df['accounttype'], errors='coerce').astype(np.float)

    df = raw_df.dropna()
    df = df[df.eval(f'scene_tmp == "{args.scene}"')]
    # df = df[df.eval(f'account_level == 4')]
    # df = df[df.eval(f'account_level == 4 and publish_level >= 1')]

    total_pv = df['pv'].sum()

    # account_df = df.groupby(['accounttype'])['pv'].sum().reset_index()
    # account_df['pv_percent'] = account_df.eval(f'pv/{total_pv}*100')
    # print(account_df)

    # df['pv_percent'] = df.eval(f'pv/{total_pv}*100')
    def func_account_level_str(x):
        if x == 1.0 or x == 2.0:
            return 'CD'
        elif x == 4.0:
            return 'A'
        else:
            return 'None'

    df['pv_percent'] = round(df.pv / total_pv * 100, 2)
    df['account_publish'] = df['account_level'].apply(func_account_level_str) + df['publish_level'].astype(str)
    df2 = df.groupby(['account_publish', 'accounttype'])['pv_percent'].sum().reset_index()

    final_df = pd.pivot(df2, 'account_publish', 'accounttype', 'pv_percent').fillna(0)
    final_df.sum(axis=1)


if __name__ == '__main__':
    # mp.set_start_method("forkserver")
    args = get_args()
    print(f'scene = {args.scene}:')

    if args.task_type in [0, 1, 2]:
        # start_date = datetime.date(2021, 1, 30)
        start_date = datetime.date(2021, 3, 1)
        # end_date = datetime.date(2021, 3, 1)
        end_date = datetime.date(2021, 3, 24)
        args.date_list = [int(d) for d in date_range(start_date, end_date)]

    print('args:', args)

    if args.task_type == 0 or args.task_type == 1:
        df = read_middle_data(args)
        df.to_csv(args.save_path)

    if args.task_type == 0 or args.task_type == 2:
        expose_trend_figure(account_level='all', publish_level_list=[1, 2, 3, 4, 5])
        # expose_trend_figure(account_level='A', publish_level_list=[1, 2, 3, 4, 5])
        # expose_trend_figure(account_level='A', publish_level_list=[2, 3, 4, 5])
        # expose_trend_figure(account_level='A', publish_level_list=[3, 4, 5])
        # expose_trend_figure(account_level='A', publish_level_list=[4, 5])
        # expose_trend_figure(account_level='A', publish_level_list=[5])
        # expose_trend_figure(account_level='CD', publish_level_list=[1, 2, 3, 4, 5])
        # expose_trend_figure(account_level='CD', publish_level_list=[0])
        # expose_trend_figure(account_level='CD', publish_level_list=[1])
        expose_trend_figure(account_level='CD', publish_level_list=[2, 3, 4, 5])
        expose_trend_figure(account_level='CD', publish_level_list=[2, 3])
        expose_trend_figure(account_level='CD', publish_level_list=[3, 4, 5])
        expose_trend_figure(account_level='CD', publish_level_list=[4, 5])
        # expose_trend_figure(account_level='all', publish_level_list=[0], y_axis_col='expose_num_all')
        # 0=0~10
        # 1=10~30
        # 2=30~50
        # 3=50~100
        # 4=100~500
        # 5=500+

    if args.task_type == 3:
        df = read_data(args=args, date=args.date)
        effect_area(df)
        # tong_compare_expose(args, args.date)

    if args.task_type == 4:
        df = read_data(args=args, date=args.date)
        effect_some_area(df)

    if args.task_type == 7:
        account_evaluate()

    if args.task_type == 8:
        run_join_expose()
