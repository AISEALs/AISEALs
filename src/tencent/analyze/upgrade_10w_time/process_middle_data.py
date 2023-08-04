import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/'
date_dir = 'short_time'
# data_type = '10-18-3day'
# data_type = '10-18-1day'
# data_type = '10-10-1day'
# data_type = '分值统计消费-2662330-20201113_2day'
# data_type = '分值统计消费-2662331-20201119_2day'
# data_type = '分值统计消费-2662859-20201113_4day'
# data_type = '分值统计消费-2662858-20201118_4day'
# data_type = '分值统计消费-2691591-20201122_3day'
# data_type = '分值统计消费-2671159-20201117_1_8day'
data_type = '分值统计消费-2730765-20201207_3day'
scene = 'video'  # 视频

# scene = 'feeds' # 图文
# data_type = '11-06-2day'

item_column = 'doc_id' if scene == 'feeds' else 'video_id'

debug_mode = False

reason_flag = False

def read_middle_data():
    file_name = os.path.join(base_dir, date_dir, f"{scene}-{data_type}.csv")
    df = pd.read_csv(file_name, sep='\t')

    # df['click'] = df['click'].apply(lambda x: int(x.split(':')[2].split('#')[1]))
    df.eval('take_hour=(ts2-ts1)/(60*60)', inplace=True)
    df.eval('speed = click/take_hour', inplace=True)
    df.eval('c2_max = 1')
    # https://stackoverflow.com/questions/15705630/get-the-rows-which-have-the-max-count-in-groups-using-groupby
    df['c2_max'] = df.groupby([item_column], sort=False)['c2'].transform(max)

    # print(df.describe())

    return df


def analyze_speed():
    # start_nums = [i for i in range(1, 12, 1)]
    #
    # c2_max_range = [(0, 5), (5, 8), (8, 10), (10, 12), (12, 100)]

    # for c2_max_start, c2_max_end in c2_max_range:
    #     print('-' * 20)
    #     print(f"final click num: [{c2_max_start}, {c2_max_end})w")
    #     c_df = df[df.eval(f"c2_max >= {c2_max_start * 10} and c2_max < {c2_max_end * 10}")]
    #     for start in start_nums:
    #         speed_df = c_df[c_df.eval(f'c2 == {start * 10}')].speed
    #
    #         print(
    #             f'{start}w: {c_df.video_id.count()}条, min_max:{"%.2f" % speed_df.min()} ~ {"%.2f" % speed_df.max()}, mean:{"%.2f" % speed_df.mean()}/hour')

    if scene == 'feeds':
        max_take_hour = 20
    else:
        max_take_hour = 50 if scene_id == 3 else 100
    ids_reach_10w_less_hour = set(df[df.eval(f'c2 == 100 and take_hour < {max_take_hour}')][item_column].values)
    ids_reach_10w_all = set(df[df.eval(f'c2 == 100')][item_column].values)

    print(f'{data_type}，新增内容{df[df.eval("c2 == 0")][item_column].count()}')
    # print(f'全部优质内容爆款耗时:\n{df[df.c2 == 100].take_hour.describe()}')
    print('-' * 20)

    # print(df[df.c2 == 100].count())
    # reach_num 单位：w
    print(f'优质内容(达到10w消费的全部内容): 一共有{len(ids_reach_10w_all)}条数据')
    print(f'优质内容(达到10w消费且{max_take_hour}h以内): 一共有{len(ids_reach_10w_less_hour)}条数据')

    reasons = [(5, 0.95), (10, 0.95), (20, 0.9), (40, 0.75), (60, 0.60)]
    reasons_range = [(0, 40), (40, 60), (60, 80), (80, 120), (120, 1000)]
    reach_10w_num_range = [1, 5] + list(range(10, 120, 10))
    not_reach_10w_num_range = [0, 1, 5] + list(range(10, 100, 10))
    for reach_num in reach_10w_num_range:
        reach_num_df = df[df.eval(f'c2 == {reach_num}')]
        # print('-' * 20)
        print(f'达到{reach_num}k消费 一共有{len(reach_num_df)}条数据, 50分位耗时：{"%.2f" % reach_num_df["take_hour"].quantile(0.5)}：')
        def test():
            old_threshold = {5: 1086, 10: 1756, 20: 2136, 40:1970, 60: 2127}
            if reach_num in old_threshold:
                speed_threholds = old_threshold.get(reach_num)
                # 到达卡点，且速度大于卡点速度
                df3 = reach_num_df[reach_num_df.speed >= speed_threholds]
                predict_ids = set(df3[item_column].values)
                precision = len(predict_ids & ids_reach_10w_less_hour) / len(predict_ids)
                recall = len(predict_ids & ids_reach_10w_less_hour) / len(ids_reach_10w_less_hour)
                # print(i, "%.2f条消费/h" % speed_threholds)
                if precision + recall > 0:
                    print(
                        f'speed {"%d" % speed_threholds}条/h, predict_num: {len(predict_ids)}, precision: {"%.2f" % precision}, recall: {"%.2f" % recall}, f1: {"%.2f" % (2 * precision * recall / (precision + recall))}')

        def analyze():
            for i in [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.98]:
                # old_threshold = {5: 0.95, 10: 0.95, 20: 0.90, 40: 0.75, 60: 0.60}
                # if reach_num not in old_threshold or old_threshold.get(reach_num) != i:
                #     continue
                speed_threholds = reach_num_df.speed.quantile(i)
                # 到达卡点，且速度大于卡点速度
                df3 = reach_num_df[reach_num_df.speed >= speed_threholds]
                predict_ids = set(df3[item_column].values)
                precision = len(predict_ids & ids_reach_10w_less_hour) / len(predict_ids)
                recall = len(predict_ids & ids_reach_10w_less_hour) / len(ids_reach_10w_less_hour)
                # print(i, "%.2f条消费/h" % speed_threholds)
                if precision + recall > 0:
                    print(
                        f'speed quantile: {"%.2f" % i}, {"%d" % speed_threholds}消费/h, predict_num: {len(predict_ids)}, precision: {"%.2f" % precision}, recall: {"%.2f" % recall}, f1: {"%.2f" % (2 * precision * recall / (precision + recall))}')

                #
                # if reach_num == 20 and i == 0.9:
                if reason_flag and (reach_num, i) in reasons:
                    # 不准的部分
                    not_in_df = df3[~df3[item_column].isin(ids_reach_10w_less_hour)]
                    # 到达10w但超时
                    reach_10_but_timeout = not_in_df[not_in_df.eval('c2_max >= 100')]
                    not_reach_10w = not_in_df[not_in_df.eval('c2_max < 100')]

                    reach_10_but_timeout_ids = set(reach_10_but_timeout[item_column].values)
                    tmp_df = df[df[item_column].isin(reach_10_but_timeout_ids)]
                    feature_df = tmp_df.pivot(index=item_column, columns='c2', values='take_hour')[reach_10w_num_range]
                    print("上一条就是选取的速度卡点，不准的部分分布如下：")
                    # print(feature_df.quantile(0.5))
                    # print(feature_df[100])
                    from collections import defaultdict
                    count = defaultdict(int)
                    for take_hour in feature_df[100]:
                        for (min, max) in reasons_range:
                            if (take_hour >= min and take_hour < max):
                                count[f"{min}_{max}"] += 1

                    not_reach_10w_ids = set(not_reach_10w[item_column].values)
                    tmp_df = df[df[item_column].isin(not_reach_10w_ids)]
                    feature_df = tmp_df.pivot(index=item_column, columns='c2', values='take_hour')[not_reach_10w_num_range]

                    print(f"到达10w但超过50h的资源，数量:{len(reach_10_but_timeout_ids)}, 没有到达10w的资源，数量:{len(not_reach_10w)}，分布如下:")
                    # print(feature_df.quantile(0.5))
                    count["0_40"] += len(feature_df)

                    total_num = len(reach_10_but_timeout_ids) + len(not_reach_10w)
                    for (min, max) in reasons_range:
                        key = f'{min}_{max}'
                        if key == '0_40':
                            print(f'没有达到10w消费的资源: {count[key]}条, 占比：{"%.2f" % (count[key] * 100.0 / total_num)}%')
                        else:
                            print(f'达到10w消费且其耗时在{key}h内: {count[key]}条, 占比：{"%.2f" % (count[key] * 100.0 / total_num)}%')

        # test()
        analyze()


def analyze_kk(df, level_range, desc):

    print('-' * 20)
    df = df[df.c2.isin(level_range)]
    if scene == 'feeds':
        df.eval('favor_percent = favor_cnt/click', inplace=True)
        df.eval('ctr = click/expose', inplace=True)
        df.eval('collect_percent = collect_cnt/click', inplace=True)
        df.eval('share_percent = share_cnt/click', inplace=True)

    title_suffix = data_type.split('-')[2]
    title = f'{title_suffix} {desc} {df[item_column].count()} {scene}'

    feature_df = df.pivot(index=item_column, columns='c2')
    for threshold in [10, 50, 90]:
        quantile_df = feature_df.quantile(threshold/100.0)
        print(f"{title_suffix} 爆款达到10w的资源，数量:{len(feature_df)}，它的{threshold}分位耗时如下：")
        print(quantile_df['take_hour'])

        aa = quantile_df['take_hour'].reset_index()
        aa.columns = ['c2', 'p']
        # aa.plot(x='p', y='c2', kind='line', marker='o', title=f'{threshold} quantile speed {title_suffix}')
        # aa.plot(x='p', y='c2', kind='line', marker='o', label = f'{threshold} label')
        plt.plot(aa['p'], aa['c2'], label=f'{threshold}quantile', marker='o')
    plt.xlabel('take time/h')
    plt.ylabel('reach xx vv/k')
    plt.title(f'{title}')
    plt.legend()
    plt.show()

    if scene == 'feeds':
        pos = 20
        quantile_df = feature_df.quantile(pos/100.0)
        print(f"爆款达到10w的资源，数量:{len(feature_df)}，它的{pos}分位ctr如下：")
        # print(feature_df['ctr'].quantile(0.10))
        print(quantile_df['ctr'])
        print(f"爆款达到10w的资源，数量:{len(feature_df)}，它的{pos}分位点赞(favor_cnt/click)如下：")
        print(quantile_df['favor_percent'])
        print(f"爆款达到10w的资源，数量:{len(feature_df)}，它的{pos}分位分享(share_cnt/click)如下：")
        print(quantile_df['share_percent'])
        print(f"爆款达到10w的资源，数量:{len(feature_df)}，它的{pos}分位收藏(collect_cnt/click)如下：")
        print(quantile_df['collect_percent'])


if __name__ == '__main__':
    df = read_middle_data()

    if scene != 'feeds':
        scene_id = 4
        df = df[df.eval(f'scene == {scene_id}')]

    print('-' * 20)

    # analyze_speed()

    # ------
    # 最后落在10w+，且时长<50h全部资源：
    # df_10w = df[df.video_id.isin(ids_reach_10w_less_hour)]  # not in 10w 50h
    # 最后落在10w + 的全部资源：
    # 非爆款 耗时：
    # df_10w = df[~df.video_id.isin(ids_reach_10w_less_hour)]
    df_1w = df[df.eval('c2_max >= 10')]  # 达到1w
    analyze_kk(df_1w, list(range(0, 11, 1)), 'reach_1w_click')
    df_10w = df[df.eval('c2_max >= 100')]  # 爆款
    reach_10w_num_range = list(range(0, 11, 1))
    analyze_kk(df_10w, reach_10w_num_range, 'reach_10w_click')
    reach_10w_num_range = [0, 1, 5] + list(range(10, 110, 10))
    analyze_kk(df_10w, reach_10w_num_range, 'reach_10w_click')


