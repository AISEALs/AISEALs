import gzip
from collections import defaultdict
import math
import pandas as pd
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_raw_data_from_online():
    samples = defaultdict(list)
    with gzip.open('/Users/jiananliu/Downloads/part-00000.gz', 'rb') as f:
        # file_content = f.read().decode('utf-8')
        for line in f:
            sp = line.decode('utf-8').split('|')
            # print(sp[2], sp[3], sp[4], sp[8])
            unique_key = f'{sp[2]}_{sp[3]}_{sp[4]}'
            d = {}
            ext_info = sp[8].strip()
            for kv in ext_info.split(';'):
                kv_sp = kv.split(':')
                if len(kv_sp) != 2:
                    continue
                k = kv_sp[0]
                v = kv_sp[1]
                d[k] = v
            cols = ['iTime', 'playTime', 'float_time_score']
            flag = True
            for col in cols:
                if col not in d:
                    flag = False
            if flag:
                for col in cols:
                    samples[unique_key].append(d[col])

    has_score = 0
    with open('/Users/jiananliu/Downloads/result1', 'r') as f:
        for lines in f:
            sp = lines.replace("Predict detail:", "").split('|')
            for line in sp:
                sp2 = line.strip().split('#')
                if len(sp2) < 2:
                    continue
                unique_key = sp2[0].strip(';')
                score = sp2[2].strip(",")
                score = sigmoid(float(score))
                if unique_key in samples:
                    samples[unique_key].append(score)
                    has_score += 1

    # print(samples)
    data = [v for _, v in samples.items()]

    print("total sample num:", len(samples))
    print("has score num:", has_score)
    return data


def gen_label(playtime, minpt):
    maxp = 240.0
    maxlogp = 8.0
    pttmp = 0
    if playtime >= minpt:
        pttmp = playtime
    pt = min(max(0.0, pttmp), maxp)
    logpt = pow(pt, 0.5)
    pctr = logpt / maxlogp
    return max(min(pctr, 0.99), 0.0)


def mse_loss(y_pred, y_label):
    return pow(y_pred - y_label, 2)


def ce_loss(y_pred, y_label):
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    loss = - y_label * np.log(y_pred) - (1 - y_label) * np.log(1 - y_pred)
    return loss


data = get_raw_data_from_online()

# print(data)
df = pd.DataFrame(data, columns=['duration', 'play_time', 'old_score', 'new_score'])
df['duration'] = df['duration'].astype(int)
df['play_time'] = df['play_time'].astype(int)
df['old_score'] = df['old_score'].astype(float)
df['new_score'] = df['new_score'].astype(float)
df['old_label'] = df['play_time'].apply(lambda x: gen_label(x, 9.0))
df['new_label'] = df['play_time'].apply(lambda x: gen_label(x, 0.0))
df['old_mse_loss'] = df.apply(lambda row: mse_loss(row['old_score'], row['old_label']), axis=1)
df['new_mse_loss'] = df.apply(lambda row: mse_loss(row['new_score'], row['new_label']), axis=1)
df['old_ce_loss'] = df.apply(lambda row: ce_loss(row['old_score'], row['old_label']), axis=1)
df['new_ce_loss'] = df.apply(lambda row: ce_loss(row['new_score'], row['new_label']), axis=1)

bins = [0, 7, 8, 9, 11, 14, 20, 25, 40, 100, 100000]
labels = [7, 8, 9, 11, 14, 20, 25, 40, 100, 100000]
# labels = ['<30', '30-60', '60-90', '>90']
# df['duration_bucket'] = pd.cut(df['duration'], bins=bins)
df['duration_bucket'] = pd.cut(df['duration'], bins=bins, labels=labels)

# 按照分桶后的duration列进行分组，并计算score列的均值
# result = df.groupby('duration_bucket')['new_score'].mean()
# result = df.groupby('duration_bucket')['new_score'].agg(['mean', 'count'])
result = df.groupby('duration_bucket').agg({'old_score': 'mean',
                                            'new_score': 'mean',
                                            'old_label': 'mean',
                                            'new_label': 'mean',
                                            'old_mse_loss': 'mean',
                                            'new_mse_loss': 'mean',
                                            'old_ce_loss': 'mean',
                                            'new_ce_loss': 'mean',
                                            'duration': 'count'})
print(result)
result2 = df.agg({
    'old_ce_loss': 'mean',
    'new_ce_loss': 'mean'
})
print(result2)
result.to_csv('final.result', sep='\t')
