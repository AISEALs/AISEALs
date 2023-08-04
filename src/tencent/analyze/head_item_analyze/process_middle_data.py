import os
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换 sans-serif 字体）
plt.rcParams['axes.unicode_minus'] = False # 步骤二（解决坐标轴负数的负号显示问题）

import seaborn as sns
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})



base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/'
test_name = 'head_item_analyze'
# ----[11-10日 ~ 11-13)  ds < 11-16
test_sql_id = 'short-test-2655645-20201119155417'


debug_mode = False
reason_flag = False
item_column = 'video_id'

def read_middle_data():
    file_name = os.path.join(base_dir, test_name, f"{test_sql_id}.csv")
    df = pd.read_csv(file_name, sep='\t')
    # video_id	c_10min	take_seconds	c_10min	click	expose	favor_cnt	comment_cnt	quality_score	account_level	hot_weight

    df = df[~df.account_level.isna()]
    return df

def analyze_speed():
    start_nums = [i for i in range(1, 12, 1)]

    c2_max_range = [(0, 5), (5, 8), (8, 10), (10, 12), (12, 100)]

    # for c2_max_start, c2_max_end in c2_max_range:
    #     print('-' * 20)
    #     print(f"final click num: [{c2_max_start}, {c2_max_end})w")
    #     c_df = df[df.eval(f"c2_max >= {c2_max_start * 10} and c2_max < {c2_max_end * 10}")]
    #     for start in start_nums:
    #         speed_df = c_df[c_df.eval(f'c2 == {start * 10}')].speed
    #
    #         print(
    #             f'{start}w: {c_df.video_id.count()}条, min_max:{"%.2f" % speed_df.min()} ~ {"%.2f" % speed_df.max()}, mean:{"%.2f" % speed_df.mean()}/hour')

    print(f'10-18号 到 10-21号 3天，新增内容{total_df[total_df.eval("c2 == 0")][item_column].count()}')
    if scene != 'feeds':
        max_take_hour = 50 if scene == 3 else 100
    else:
        max_take_hour = 20
    print(f'全部优质内容爆款耗时:\n{total_df[total_df.c2 == 100].take_hour.describe()}')
    print('-' * 20)

    # print(df[df.c2 == 100].count())
    ids_reach_10w_less_hour = set(total_df[total_df.eval(f'c2 == 100 and take_hour < {max_take_hour}')][item_column].values)
    ids_reach_10w_all = set(total_df[total_df.eval(f'c2 == 100')][item_column].values)

    # reach_num 单位：w
    print(f'优质内容(达到10w消费的全部内容: 一共有{len(ids_reach_10w_all)}条数据')
    print(f'优质内容(达到10w消费且{max_take_hour}h以内): 一共有{len(ids_reach_10w_less_hour)}条数据')

    reasons = [(5, 0.95), (10, 0.95), (20, 0.9), (40, 0.75), (60, 0.60)]
    reasons_range = [(0, 40), (40, 60), (60, 80), (80, 120), (120, 1000)]
    reach_10w_num_range = [1, 5] + list(range(10, 120, 10))
    not_reach_10w_num_range = [0, 1, 5] + list(range(10, 100, 10))
    for reach_num in reach_10w_num_range:
        reach_num_df = total_df[total_df.eval(f'c2 == {reach_num}')]
        print('-' * 20)
        print(f'达到{reach_num}k消费 一共有{len(reach_num_df)}条数据：')
        for i in [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.98]:
            speed_threholds = reach_num_df.speed.quantile(i)
            # 到达卡点，且速度大于卡点速度
            df3 = reach_num_df[reach_num_df.speed >= speed_threholds]
            predict_ids = set(df3[item_column].values)
            precision = len(predict_ids & ids_reach_10w_less_hour) / len(predict_ids)
            recall = len(predict_ids & ids_reach_10w_less_hour) / len(ids_reach_10w_less_hour)
            # print(i, "%.2f条消费/h" % speed_threholds)
            if precision + recall > 0:
                print(
                    f'speed quantile: {"%.2f" % i}, {"%d" % speed_threholds}条/h, predict_num: {len(predict_ids)}, precision: {"%.2f" % precision}, recall: {"%.2f" % recall}, f1: {"%.2f" % (2 * precision * recall / (precision + recall))}')
            # print(f'分位耗时：{"%.2f" % df3.take_hour.quantile(i)}h, 计算耗时: {"%.2f" % (1000.0*reach_num/int(speed_threholds))}, 流量速率:{speed_threholds}/h')

            #
            # if reach_num == 20 and i == 0.9:
            if reason_flag and (reach_num, i) in reasons:
                # 不准的部分
                not_in_df = df3[~df3[item_column].isin(ids_reach_10w_less_hour)]
                # 到达10w但超时
                reach_10_but_timeout = not_in_df[not_in_df.eval('c2_max >= 100')]
                not_reach_10w = not_in_df[not_in_df.eval('c2_max < 100')]

                reach_10_but_timeout_ids = set(reach_10_but_timeout[item_column].values)
                tmp_df = total_df[total_df[item_column].isin(reach_10_but_timeout_ids)]
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
                tmp_df = total_df[total_df[item_column].isin(not_reach_10w_ids)]
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


if __name__ == '__main__':
    total_df = read_middle_data()
    # total_df = total_df[total_df.eval('expose >= click and click >= favor_cnt')]
    total_df.eval('ctr = click*1.0/(expose+1)', inplace=True)
    total_df.eval('favor_percent = favor_cnt*1.0/(click+20)', inplace=True)

    # 72/6=12hour
    df_12h = total_df[total_df.eval('interval_time == 12')]
    print(df_12h['expose'].describe())
    print(df_12h['favor_cnt'].describe())

    print('-'*20)
    quantile_80_value = df_12h['expose'].quantile(0.8)
    print(f"曝光数量80分位值：{quantile_80_value}")

    expose_q80_df = df_12h[df_12h.eval(f"expose > {quantile_80_value}")]
    expose_q80_ids = set(expose_q80_df['video_id'].values)

    # favor_cnt_q80 = df_12h.favor_cnt.quantile(0.9)
    favor_percnet_q80 = df_12h.favor_percent.quantile(0.9)
    # (df_12h.favor_cnt > favor_cnt_q80) &
    favor_high = df_12h[df_12h.favor_percent > favor_percnet_q80]

    total_high_num = favor_high['video_id'].count()
    print("总数量")
    step = 0.2
    for start in np.arange(0, 1, step):
        end = start + step
        q_start_v = df_12h['expose'].quantile(start)
        q_end_v = df_12h['expose'].quantile(end)
        tmp_df = favor_high[favor_high.eval(f"expose >= {q_start_v} and expose < {q_end_v}")]
        tmp_num = tmp_df['video_id'].count()
        print(f'{start}~{end}: ({q_start_v, q_end_v}){"%.2f" % (tmp_num*1.0/total_high_num)}, {tmp_num}, ')


    favor_high_ids = set(favor_high['video_id'].values)
    not_favor_ids = favor_high_ids - expose_q80_ids
    not_df = df_12h[df_12h.video_id.isin(not_favor_ids)]
    print(f'头部曝光(80分位以上)数量：{len(expose_q80_ids)}\n头部点赞(80分位以上)资源量：{len(favor_high_ids)}\n其中{len(not_favor_ids)}不再头部曝光里，占{"%.2f" % (len(not_favor_ids)*1.0/len(favor_high_ids))}')
    print(not_df['ctr'].describe())
    expose_min = not_df.expose.min()
    expose_max = not_df.expose.max()
    # vs not_df
    other_expose_df = df_12h[df_12h.eval(f'expose >= {expose_min} and expose <= {expose_max}')]
    other_expose_df = other_expose_df[~other_expose_df.video_id.isin(not_favor_ids)]
    print(f'最小曝光：{expose_min}, 最大曝光：{expose_max}, 对比全局相同曝光位置资源：{other_expose_df.video_id.count()}')
    print(f'这部分资源{len(not_favor_ids)}点击率：')
    print(not_df.ctr.mean())
    # print(not_df['favor_percent'].describe())
    print(f'同位曝光的全局资源（已除去上面的{len(not_favor_ids)}条)点击率：')
    print(other_expose_df.ctr.mean())
    # print(other_expose_df['favor_percent'].describe())
    print('-' * 20)


    def plot_expose(df):
        aa = []
        for i in np.arange(0.1, 1, 0.1):
            q = df['expose'].quantile(i)
            aa.append((int(i * 100), q))
        q_df = pd.DataFrame(aa, columns=['quantile', 'expose'])

        sns.barplot(x='quantile', y='expose', data=q_df)

    # plot_expose(df_12h)
    # plt.show()
    # plot_expose(favor_high)
    #
    # plt.show()

