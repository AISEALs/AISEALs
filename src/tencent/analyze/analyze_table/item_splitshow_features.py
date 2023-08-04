import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/table'
sub_dir_name = 'item_split_grow'

scene = 'mini_video'  # 视频
# scene = 'short_video'  # 视频
# scene = 'feeds' # 图文
is_quality_account = False
is_baokuan = False

item_column = 'item_id'

global date_label


def read_middle_data(data_type, idx_list):
    global date_label
    if isinstance(idx_list, list):
        df_list = []
        for dateStr in idx_list:
            file_name = os.path.join(base_dir, sub_dir_name, f"item_split_grow-{data_type}-{dateStr}.csv")
            df_list.append(pd.read_csv(file_name, sep='\t'))
        df = pd.concat(df_list, axis=0, ignore_index=True)
        date_label = f'{len(idx_list)}day'
    else:
        file_name = os.path.join(base_dir, sub_dir_name, f"item_split_grow-{data_type}-{idx_list}.csv")
        df = pd.read_csv(file_name, sep='\t')
        date_label = '1day'


    df['c2'] = df['view_level']

    if scene == 'mini_video':
        df = df[df.eval('biz_type == 19')]
    elif scene == 'short_video':
        df = df[df.eval('biz_type == 0 or biz_type == 3')]
    else:
        raise Exception('not support feeds')
    # SAB C D = 543 2 1
    if is_quality_account:
        df = df[df.eval('account_level >= 3')]

    print(f'is quality: {is_quality_account}')
    print(f'{scene} {date_label} total num:{df[item_column].count()}, distinct item num: {df[item_column].nunique()}')

    df.eval('take_hour=(ts2-ts1)/(60*60)', inplace=True)
    df.eval('ctr = click/(expose+1)', inplace=True)
    # df.eval('speed = click/take_hour', inplace=True)
    # https://stackoverflow.com/questions/15705630/get-the-rows-which-have-the-max-count-in-groups-using-groupby
    # df = df[df.eval('item_id == 2696524497611541882')] # for test
    df_max = df.groupby([item_column], sort=False).transform(max)
    df['c2_max'] = df_max['view_level']
    df['click_max'] = df_max['max_click_in_level']
    df['expose_max'] = df_max['max_expose_in_level']
    df_min = df.groupby([item_column], sort=False).transform(min)
    df['expose_min'] = df_min['expose']

    return df


def test_click_vs_expose_10wclick_is_same():
    # for test
    click_df = read_middle_data('click', [20201220, 20201221])

    ids1 = set(click_df[click_df.eval('c2 >= 1000')]['item_id'])
    ids2 = set(click_df[click_df.eval('click >= 100000')]['item_id'])
    ids3 = set(df[df.eval('click >= 100000')]['item_id'])
    assert ids1 == ids2 == ids3

def filter():
    # if scene == 'feeds':
    #     df.eval('favor_percent = favor_cnt/click', inplace=True)
    #     df.eval('ctr = click/expose', inplace=True)
    #     df.eval('collect_percent = collect_cnt/click', inplace=True)
    #     df.eval('share_percent = share_cnt/click', inplace=True)
    # if scene == 'feeds':
    #     pos = 20
    #     quantile_df = feature_df.quantile(pos/100.0)
    #     print(f"爆款达到10w的资源，数量:{len(feature_df)}，它的{pos}分位ctr如下：")
    #     # print(feature_df['ctr'].quantile(0.10))
    #     print(quantile_df['ctr'])
    #     print(f"爆款达到10w的资源，数量:{len(feature_df)}，它的{pos}分位点赞(favor_cnt/click)如下：")
    #     print(quantile_df['favor_percent'])
    #     print(f"爆款达到10w的资源，数量:{len(feature_df)}，它的{pos}分位分享(share_cnt/click)如下：")
    #     print(quantile_df['share_percent'])
    #     print(f"爆款达到10w的资源，数量:{len(feature_df)}，它的{pos}分位收藏(collect_cnt/click)如下：")
    #     print(quantile_df['collect_percent'])
    pass


def item_grow_figure(df, desc, show=False):
    print('-' * 20)
    title = f'{date_label} {desc} {len(set(df[item_column]))} {scene}'

    feature_df = df.pivot(index=item_column, columns='c2')
    for threshold in [10, 50, 70, 90]:
        quantile_df = feature_df.quantile(threshold/100.0)
        print(f"{title}，它的{threshold}分位耗时如下：")
        print(quantile_df['take_hour'])

        q_take_hour_df = quantile_df['take_hour'].reset_index()
        q_take_hour_df.columns = ['c2', 'p']
        # aa.plot(x='p', y='c2', kind='line', marker='o', title=f'{threshold} quantile speed {title_suffix}')
        plt.plot(q_take_hour_df['p'], q_take_hour_df['c2'], label=f'{threshold}quantile', marker='o')
    plt.xlabel('take time/h')
    plt.ylabel(f'reach xx {data_type}/100')
    plt.title(f'{title}')
    plt.legend()
    if show:
        plt.show()

def expose_level_figure(df):
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    # plt.figure(1)
    df_1k = df[df.eval('c2_max >= 10')]  # 达到1k
    df_1k = df_1k[df_1k.c2.isin(list(range(0, 11, 1)))]
    item_grow_figure(df_1k, f'reach_1k_{data_type}')

    plt.subplot(1, 3, 2)
    df_1w = df[df.eval('c2_max >= 100')]  # 达到1w
    df_1w = df_1w[df_1w.c2.isin(list(range(0, 10, 2)) + list(range(0, 110, 10)))]
    item_grow_figure(df_1w, f'reach_1w_{data_type}')

    plt.subplot(1, 3, 3)
    df_10w = df[df.eval('c2_max >= 1000')]  # 达到10w曝光
    df_10w = df_10w[df_10w.c2.isin(list(range(0, 100, 10)) + list(range(0, 1100, 100)))]
    item_grow_figure(df_10w, f'reach_10w_{data_type}')

    plt.tight_layout()
    plt.show()
    plt.savefig('example_6_2.png')
    plt.close()

def ctr_trend_figure(df):
    print('-' * 20)
    reach_expose = 1000
    title = f'ctr trend(reach {int(reach_expose/100)}w expose doc)'

    item_ids = df[df.eval(f'c2_max == {reach_expose}')]['item_id']
    item_ids = np.random.choice(item_ids, 5)

    for i in range(len(item_ids)):
        item_id = item_ids[i]
        feature_df = df
        feature_df = feature_df[feature_df.eval(f'item_id == {item_id}')]
        if reach_expose >= 1000:
            reach_expose = 100
        feature_df = feature_df[feature_df.eval(f'c2 <= {reach_expose} and c2 > 0')]

        q_take_hour_df = feature_df[['c2', 'ctr']].sort_values(by='c2')
        plt.plot(q_take_hour_df['c2'], q_take_hour_df['ctr'], label=f'doc{i}', marker='o')
    plt.xlabel(f'reach xx {data_type}/100')
    plt.ylabel('ctr')
    plt.title(f'{title}')
    plt.legend()
    plt.show()

def expose_dist_figure(df):
    ii = np.arange(0, 100, 1)
    x = [df['expose_max'].quantile(i/100)/10000 for i in ii]
    for i in list(np.arange(0, 90, 10)) + list(np.arange(90, 101, 2)):
        print(f'|q{i}|{int(df["expose_max"].quantile(i/100.0))}|{int(df["click_max"].quantile(i/100.0))}|')

    plt.bar(ii, x, 0.5)
    plt.xlabel('quantile')
    plt.ylabel('expose num/w')
    plt.title(f'{len(set(df[item_column]))} {scene} expose dist')
    plt.show()

def compare_expore_figure(tail2df):
    # plt.figure(figsize=(30, 10))
    # plt.subplot(1, 3, 1)
    # plt.figure(1)
    desc = f'reach_1k_{data_type}'
    title = f'{date_label} {desc} {scene}'
    for tail, df in tail2df.items():
        df_1k = df[df.eval('c2_max >= 100')]  # 达到1k
        df_1k = df_1k[df_1k.c2.isin(list(range(0, 11, 1)))]
        # item_grow_figure(df_1k, f'reach_1k_{data_type}')
        print('-' * 20)

        feature_df = df_1k.pivot(index=item_column, columns='c2')
        for threshold in [10, 50]:
            quantile_df = feature_df.quantile(threshold/100.0)
            print(f"{title}, tail:{tail}，它的{threshold}分位耗时如下：")
            print(quantile_df['take_hour'])

            q_take_hour_df = quantile_df['take_hour'].reset_index()
            q_take_hour_df.columns = ['c2', 'p']
            plt.plot(q_take_hour_df['p'], q_take_hour_df['c2'], label=f'tail:{tail} {threshold}quantile', marker='o')
    plt.xlabel('take time/h')
    plt.ylabel(f'reach xx {data_type}/100')
    plt.title(f'{title}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def test_hash(df):
    df = df[df.eval('c2_max >= 80')]  # 达到1k
    df = df[df.eval("c2 == c2_max")]
    print(df.groupby('tail_num')['item_id'].count())

    for tail in list(range(0, 10)):
        tmp_df = df[df.eval(f'tail_num == "{tail}"')]
        total_expose_num = tmp_df.eval("expose_max - expose_min").sum()
        print(
            f'tail: {tail}, item num: {tmp_df[item_column].count()}, total expose num: {total_expose_num}, mean expose:{total_expose_num / (tmp_df[item_column].count())}')


if __name__ == '__main__':
    data_type = 'expose'
    # data_type = 'click'

    df = read_middle_data(data_type, [20210106, 20210107, 20210108, 20210109])
    # df = read_middle_data(data_type, [20201219, 20201220, 20201221])

    last_df = df[df.eval('c2 == c2_max')]
    assert last_df['item_id'].count() == len(set(df['item_id'].values))

    expose_dist_figure(df)

    # ctr_trend_figure(df)

    print('-' * 20)

    # 爆款资源（达到10w消费），曝光成长图
    if is_baokuan:
        df = df[df.eval(f'click_max >= 100000')]

    # expose_level_figure(gray_df)
    gray_id = 7
    if gray_id != -1:
        # 编码方式：判断docID倒数第三位，若为2，则曝光增大为500；若为7，则曝光增大为1000
        df.loc[:, 'tail_num'] = df['item_id'].apply(lambda x: str(x)[-3:-2])
        tails = [1, 2, 3, 7]
        # tails = list(range(0, 10))
        tail2df = {tail: df[df.eval(f'tail_num == "{tail}"')] for tail in tails}
        compare_expore_figure(tail2df)

    # expose_level_figure(df)

    test_hash(df)
