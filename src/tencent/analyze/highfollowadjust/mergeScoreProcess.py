import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import mmap
import matplotlib.pyplot as plt
import traceback
from itertools import chain


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_df():
    file_name = os.path.join('/src/tencent/analyze/highfollowadjust/data/test.log')
    lines = []
    debug = False
    num = 100000 if debug else get_num_lines(file_path=file_name)
    with open(file_name, 'r') as f:
        for line_num in tqdm(range(num)):
            try:
                line = f.__next__()
                sp = line.split('|')
                fields = dict()
                for i in sp:
                    ssp = i.split(':')
                    if len(ssp) == 2:
                        key = ssp[0].strip()
                        value = ssp[1].strip()
                        if 'score' in key or 'ratio' in key or 'offset' in key or 'pow' in key:
                            value = float(value)
                        if key in ['merge_method', 'exp_type', 'isPE', 'deepFollowUserThreshold',
                                   'bFollowStrengthenFlag', 'followAccountsNum']:
                            value = int(value)
                        if key in ['merge_param1716', 'deepFollowUserRatio', 'normalFollowUserRatio',
                                   'followedDecayWeight', 'dMTFollowScore']:
                            value = float(value)
                        if len(key) > 100:
                            continue
                        if key == 'followAccountsNum' and value < 0:
                            continue
                        fields[key] = value
                if len(fields) >= 38:
                    # lines.append(list(fields.values()))
                    lines.append(fields)
            except Exception as ex:
                print(line_num)
                traceback.print_exc()

    # tqdm.pandas(desc='my bar!')
    print(sys.getsizeof(lines))
    df = pd.DataFrame.from_records(lines)
    print(sys.getsizeof(df))
    print(df.info())
    return df


def show_follow_account_info(df):
    df['followAccountsNum'].describe()
    q_indexes = list(np.arange(0, 1.0, 0.1))
    q_indexes.extend(np.arange(0.9, 1.0, 0.02))
    q_df = df['followAccountsNum'].quantile([i for i in q_indexes])
    print('-' * 20)
    print(f'follow account num quantile: {q_df}')
    print('-' * 20)
    q_df.plot(marker='o', title='follow account distribution')
    plt.show()


def show_follow_obj_effect(df):
    def f(gray_id, ratio_, offset_, pow_, rescale, deepFollowUserThreshold=0, deepFollowRatio=0, followedDecayWeight=0):
        result = []
        # def cal_new_follow_ratio(follow_num):
        #     if (follow_num > deepFollowUserThreshold):
        #         return deepFollowRatio*followedDecayWeight
        #     else:
        #         return 0
        # df['fnewFollowRatio'] = df['followAccountsNum'].apply(cal_new_follow_ratio)
        if rescale:
            pow_of_follow_score_df = pow((df.ori_follow_score_ * ratio_ + offset_), pow_)
        else:
            pow_of_follow_score_df = pow(df.dMTFollowScore, pow_)
        q_indexes = list(np.arange(0, 1.0, 0.1))
        q_indexes.extend(np.arange(0.9, 1.0, 0.02))
        name = f'{gray_id}: pow({ratio_}x+{offset_}, {pow_})'
        q_follow_score = pow_of_follow_score_df.quantile([i for i in q_indexes])
        result.append((name, q_follow_score))

        pow_of_mtexceptfollow_df = pow(df.MTscore, df.merge_param1716)
        # q_mtexceptfollow_score = pow_of_mtexceptfollow_df.quantile([i for i in q_indexes])
        # # # result['mtExceptFollow'] = q_mtexceptfollow_score
        # # result.append(('mtExceptFollow', q_mtexceptfollow_score))

        # pow_of_mt_score_df = df.ctr_score * pow_of_mtexceptfollow_df * (1.0/df.ctr_score.quantile(0.5))
        pow_of_mt_score_df = pow_of_mtexceptfollow_df
        q_mtscore_df = pow_of_mt_score_df.quantile([i for i in q_indexes])
        result.append(('otherMTScore', q_mtscore_df))

        return result

    params_list = [
        # ('base', 35, 0.01, 0.3, True),
        ('shallow', 35, 0.001, 0.3, True),
        ('deep', 35, 0.001, 0.5, True),
        # ('test1', 35, 1, 0.4, True),
        # ('test2', 35, 1, 0.6, True),
        # ('test3', 35, 1, 0.8, True),
        # (35, 0.2, 1.35)
    ]
    aa = dict(list(chain.from_iterable([f(*params) for params in params_list])))
    dd = pd.DataFrame(aa)

    dd.plot(marker='o')
    plt.xlabel('rank of xxx score quantile')
    plt.ylabel('score')
    plt.show()


def show_follow_obj_effect2(df):
    def f(df, gray_id, ratio_, offset_, pow_, rescale):
        result = []
        if rescale:
            pow_of_follow_score_df = pow((df1.ori_follow_score_ * ratio_ + offset_), pow_)
        else:
            pow_of_follow_score_df = pow(df.dMTFollowScore, pow_)
        q_indexes = list(np.arange(0, 1.0, 0.1))
        q_indexes.extend(np.arange(0.9, 1.0, 0.02))
        name = f'{gray_id}: pow({ratio_}x+{offset_}, {pow_})'
        q_follow_score = pow_of_follow_score_df.quantile([i for i in q_indexes])
        print(q_follow_score)
        result.append((name, q_follow_score))

        return result

    df1 = df[df['follow power'] == 0.3]
    df2 = df[df['follow power'] == 0.5]

    params_list = [
        ('shallow', 35, 0.001, 0.3, True),
        ('deep', 35, 0.001, 0.5, True)
    ]
    aa = dict(list(chain.from_iterable([f(df1, *params_list[0]) + f(df2, *params_list[1])])))
    dd = pd.DataFrame(aa)

    dd.plot(marker='o', title='pow of follow score distribution')
    plt.xlabel('pow of follow score quantile')
    plt.ylabel('pow of follow score')
    plt.show()


def show_video_time_by_followscore(df):
    vids = set(df.vid.values)
    # print(','.join(vids))

    doc_df = pd.read_csv('/src/tencent/analyze/highfollowadjust/data/doc.csv', sep='\t')

    df['vid'] = df['vid'].astype(np.int64)
    doc_df['finish_rate'] = doc_df['finish_rate'].astype(float)
    joined_df = df.merge(doc_df, left_on='vid', right_on='item_id', how='left')
    joined_df = joined_df.dropna()
    df3 = joined_df.groupby(pd.cut(joined_df.ori_follow_score_, 4))[['video_time', 'finish_rate']].mean()
    # df3 = joined_df.groupby(pd.cut(joined_df.ori_follow_score_, 4))['finish_rate'].mean()
    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # ax2 = ax1.twinx()
    # width = 0.3
    # df3.video_time.plot(kind='bar', color='red', ax=ax1, width=width, position=1)
    # df3.finish_rate.plot(kind='bar', color='green', ax=ax2, width=width, position=0)
    # ax1.set_ylabel('video-time/s')
    # # ax1.set_xlabel('video time')
    # ax2.set_ylabel('finish-rate')
    # # ax2.set_xlabel('finish rate')
    df3.plot(kind='bar', secondary_y='finish_rate', rot=0)
    plt.xlabel('follow-score')
    # plt.xlabel('pow of follow score quantile')
    plt.show()



if __name__ == '__main__':
    df = get_df()
    print(df.merge_param1716.value_counts())

    mean_score_list = []
    mean_ratio_fields = []
    # MTscore = praise_score + share_score + wtcomment_score + rdcomment_score + follow_score
    df.MTscore.describe()
    score_fields = ['praise_score', 'share_score', 'wtcomment_score', 'rdcomment_score', 'ori_follow_score_']
    for score_field in score_fields:
        print(f'{score_field} mean: {df[score_field].mean()}')
        mean_score_list.append(df[score_field].mean())
        print('==========')
        print(df[score_field].groupby(pd.cut(df[score_field], [i for i in np.arange(0, 1.1, 0.1)])).count())
        # print(df[score_field].groupby(pd.cut(df[score_field], [i for i in np.arange(0, 0.11, 0.01)])).count())

    # praise ratio:1.6, share ratio:5.0, write comment ratio:12.0, read comment ratio: 0.04
    # follow ratio: {20.0:99%, 16.0:1%}
    ratio_fields = ['praise ratio', 'share ratio', 'write comment ratio', 'read comment ratio', 'follow ratio']
    for ratio_field in ratio_fields:
        print(df[ratio_field].value_counts())
        mean_ratio_fields.append(df[ratio_field].mean())

    sum_of_mean_mtscore = sum(ratio * score for ratio, score in zip(mean_ratio_fields, mean_score_list))
    mean_of_mtscore = df.MTscore.mean()
    print(f'sum of mean_mtscore: {sum_of_mean_mtscore}')
    print(f'mean of mtscore: {mean_of_mtscore}')

    for field, ratio, score in zip(score_fields, mean_ratio_fields, mean_score_list):
        print(
            f'{field}: ratio:{"{:.2f}".format(ratio)}, score:{"{:.2f}".format(score)}, ratio*score={"{:.2f}".format(ratio * score)}, percent:{"{:.2%}".format(ratio * score / sum_of_mean_mtscore)}')

    show_follow_account_info(df)

    # deep_df = df[df.deepFollowUserThreshold == 4]
    # df1 = df[df.bFollowStrengthenFlag == 0]
    df2 = df[df.bFollowStrengthenFlag == 1]

    show_follow_obj_effect(df2)
    # show_follow_obj_effect2(df2)

    # show_video_time_by_followscore(df)


