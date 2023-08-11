#!/usr/bin/env python
# coding=utf-8
"""
   #### Moudle : QB Feeds click sequence data
   #### Date :  2022-03-25
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, datetime, random, time
import sklearn
from pyspark import SparkConf
from pyspark.sql import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument('--trs_sample_file_ori', type=str, default="", help='样本路径')
parser.add_argument('--date', type=str, default="", help='取date时间的数据')
parser.add_argument('--max_seq_len', type=int, default=50, help='最大序列长度')
parser.add_argument('--slot_id', type=int, default=12, help='序列特征在无量样本的SLOTID')
parser.add_argument('--sparse_slot_id', type=int, default=12, help='稀疏输出在无量样本的SLOTID')
parser.add_argument('--weight_slot_id', type=int, default=12, help='权重在无量样本的SLOTID')
parser.add_argument('--miniv_slot_id', type=int, default=12, help='小视频序列特征在无量样本的SLOTID')
parser.add_argument('--shortv_slot_id', type=int, default=12, help='短视频序列特征在无量样本的SLOTID')
parser.add_argument('--ft_miniv_slot_id', type=int, default=12, help='浮层小视频序列特征在无量样本的SLOTID')
parser.add_argument('--ft_shortv_slot_id', type=int, default=12, help='浮层短视频序列特征在无量样本的SLOTID')
parser.add_argument('--ft_pos_play_ratio', type=float, default=0.8, help='浮层刷选正例的播放率阀值')
parser.add_argument('--item_num', type=int, default=12, help='选择top-item个数')
parser.add_argument('--max_hours', type=int, default=12, help='读取数据最多的小时数')
args, _ = parser.parse_known_args()
print('args.trs_sample_file_ori:', args.trs_sample_file_ori)
print('args.date:', args.date)
print('args.max_seq_len:', args.max_seq_len)
print('args.slot_id:', args.slot_id)
print('args.sparse_slot_id:', args.sparse_slot_id)
print('args.miniv_slot_id:', args.miniv_slot_id)
print('args.shortv_slot_id:', args.shortv_slot_id)
print('args.ft_miniv_slot_id:', args.ft_miniv_slot_id)
print('args.ft_shortv_slot_id:', args.ft_shortv_slot_id)
print('args.ft_pos_play_ratio:', args.ft_pos_play_ratio)
print('args.weight_slot_id:', args.weight_slot_id)
print('args.item_num:', args.item_num)
print('args.max_hours:', args.max_hours)

date = args.date
max_seq_len = args.max_seq_len
SLOT_ID = args.slot_id
MINIV_SLOT_ID = args.miniv_slot_id
SHORTV_SLOT_ID = args.shortv_slot_id
FT_MINIV_SLOT_ID = args.ft_miniv_slot_id
FT_SHORTV_SLOT_ID = args.ft_shortv_slot_id
SPARSE_FIX_OUTPUT_SLOT_ID = args.sparse_slot_id
WEIGHT_SLOT_ID = args.weight_slot_id
FT_POS_PLAY_RATIO = args.ft_pos_play_ratio
item_num = args.item_num

numerous_date_format = "%s-%s-%s/%s" % (date[:4], date[4:6], date[6:8], date[8:])


def check_if_exist(path):
    sc_l = SparkSession.builder.getOrCreate().sparkContext
    Path = sc_l._gateway.jvm.org.apache.hadoop.fs.Path
    URI = sc_l._gateway.jvm.java.net.URI
    FileSystem = sc_l._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fs = FileSystem.get(URI(path), sc_l._jsc.hadoopConfiguration())
    if fs.exists(Path(path)):
        return True
    return False


input_file = []
# 对指定count_time(48h)内的样本进行并集操作
for i in range(args.max_hours):
    input_time = datetime.datetime.strptime(date, "%Y%m%d%H") - datetime.timedelta(hours=i)
    time_path = str(input_time).split(" ")[0] + "/" + str(input_time).split(" ")[1].split(":")[0]
    full_time_path = "{ori_path}/{time_path}".format(ori_path=args.trs_sample_file_ori, time_path=time_path)
    if not check_if_exist(full_time_path):
        print("not_full_time_path:{}".format(full_time_path))
        continue
    input_file.append(full_time_path)

input_path = ",".join(input_file)


def delete_if_exist(path):
    sc_l = SparkSession.builder.getOrCreate().sparkContext
    Path = sc_l._gateway.jvm.org.apache.hadoop.fs.Path
    URI = sc_l._gateway.jvm.java.net.URI
    FileSystem = sc_l._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fs = FileSystem.get(URI(path), sc_l._jsc.hadoopConfiguration())
    if fs.exists(Path(path)):
        print('fs.delete(Path({0}), True)'.format(path))
        fs.delete(Path(path), True)


def hash_func(key, slot_id):
    hash_key = str(slot_id) + "#" + key;
    hash_sign = sklearn.utils.murmurhash3_32(hash_key, 1540483477, True)
    hash_sign = ((slot_id << 32) & 0xfff00000000) | hash_sign
    return hash_sign


def extract_seq_with_full_vid(line):
    guid, query_id, float_short_dict, float_mini_dict = line
    ds = datetime.datetime.fromtimestamp(int(query_id)).strftime("%Y%m%d%H")
    vid_list = []
    new_guid = "%s_%s" % (guid, ds)  # every ds output a seq

    if not float_short_dict["20"] and not float_mini_dict["30"]:
        return (new_guid, vid_list)

    for i in range(len(float_short_dict["20"][:max_seq_len])):
        real_play_ratio = float(float_short_dict["21"][i])
        vid = float_short_dict["20"][i]
        vid_list.append((vid, int(float_short_dict["22"][i]), "0", max(1, real_play_ratio + 1 - FT_POS_PLAY_RATIO)))

    for i in range(len(float_mini_dict["30"][:max_seq_len])):
        real_play_ratio = float(float_mini_dict["31"][i])
        vid = float_mini_dict["30"][i]
        vid_list.append((vid, int(float_mini_dict["32"][i]), "19", max(1, real_play_ratio + 1 - FT_POS_PLAY_RATIO)))

    return (new_guid, vid_list)


def numerous_train_sample_format(guid, vid_list, mini_float_vids_index):
    vid_map = {}
    uniq_ft_mini_vid_list = []
    uniq_ft_short_vid_list = []
    samples = []

    if not vid_list:
        return []

    # (vid, action_time, video_type, max(1, real_play_ratio+0.7))
    sorted_vid_list = sorted(vid_list, key=lambda x: x[1], reverse=True)

    test_vid = sorted_vid_list[0][0]
    if random.random() < 0.01:
        return []

    for vid, _, video_type, weight in sorted_vid_list:
        k = vid
        if k in vid_map:
            continue
        if int(video_type) == 19:
            uniq_ft_mini_vid_list.append((vid, weight))
        elif int(video_type) == 0:
            uniq_ft_short_vid_list.append((vid, weight))

        vid_map[k] = 1

    # float sparse
    ft_mini_s = ["%s:%s:1" % (hash_func(vid, FT_MINIV_SLOT_ID), FT_MINIV_SLOT_ID) for vid, _ in
                 uniq_ft_mini_vid_list[:max_seq_len]]
    ft_short_s = ["%s:%s:1" % (hash_func(vid, FT_SHORTV_SLOT_ID), FT_SHORTV_SLOT_ID) for vid, _ in
                  uniq_ft_short_vid_list[:max_seq_len]]
    ft_sparse_sample = ";".join(ft_mini_s + ft_short_s)
    ft_mini_sparse_sample = ";".join(
        ["%s:%s:1" % (hash_func(vid, MINIV_SLOT_ID), SPARSE_FIX_OUTPUT_SLOT_ID) for vid, _ in
         uniq_ft_mini_vid_list[:max_seq_len] if vid in mini_float_vids_index.value])

    # float dense
    ft_mini_vid_ind_list = [(mini_float_vids_index.value[vid], w) for vid, w in uniq_ft_mini_vid_list[:max_seq_len] if
                            vid in mini_float_vids_index.value]
    ft_mini_dense_sample = ";".join(["%s:%s:%s" % (i + 1, SLOT_ID, vid) for i, (vid, w) in
                                     enumerate(ft_mini_vid_ind_list)])  # key low-bound=1  high-bound=max_seq_len
    ft_mini_weight_sample = ";".join(
        ["%s:%s:%s" % (i + 1, WEIGHT_SLOT_ID, w) for i, (vid, w) in enumerate(ft_mini_vid_ind_list)])

    ft_mini_sparse_sample = "guid:%s;vid:%s|1.0|%s;%s;%s;%s" % (
    guid, test_vid, ft_sparse_sample, ft_mini_dense_sample, ft_mini_sparse_sample, ft_mini_weight_sample)

    if len(ft_mini_vid_ind_list) > 0:
        samples.append(ft_mini_sparse_sample)

    return samples


def extract_seq(data):
    guid, query_id, features = data
    feat_list = features.split(";")
    float_short_dict = {
        "20": [],
        "21": [],
        "22": []
    }

    float_mini_dict = {
        "10": [],
        "30": [],
        "31": [],
        "32": []
    }

    for feat in feat_list:
        key, slot, value = feat.split(":")
        if slot in float_short_dict:
            float_short_dict[slot].append(value)
        if slot in float_mini_dict:
            float_mini_dict[slot].append(value)

    return guid, query_id, float_short_dict, float_mini_dict


def process_vid_sample(click_seq):
    # 过滤长度小于6的点击序列、取最新的3个点击历史、过滤被点击次数小于100的video
    float_mini_click_seq = click_seq.map(lambda x: x[3]["10"]) \
        .filter(lambda x: len(x) >= 6) \
        .flatMap(lambda x: x[:3]) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] > 100) \
        .takeOrdered(item_num, key=lambda x: -x[1])

    print("float_mini length: ", len(float_mini_click_seq))

    float_mini_index = {}
    for i, (vid, cnt) in enumerate(float_mini_click_seq):
        float_mini_index[vid] = i

    float_mini_dense_path = "mdfs://cloudhdfs/mttsparknew/data/video/trs/float_vae/item_embedding_sample/ft_miniv_dense_vid/%s" % numerous_date_format
    delete_if_exist(float_mini_dense_path)

    # sc.parallelize(float_mini_click_seq, numSlices=1) \
    #     .map(lambda x: "vid:%s;vcnt:%s|1.0|1:%s:%s"%(x[0], x[1], SLOT_ID, float_mini_index[x[0]])) \
    #     .saveAsTextFile(float_mini_dense_path)

    sc.parallelize(float_mini_click_seq, numSlices=1) \
        .map(lambda x: "%s|1.0|1:%s:%s" % (x[0], SLOT_ID, float_mini_index[x[0]])) \
        .saveAsTextFile(float_mini_dense_path)

    return float_mini_index


def process_train_sample(click_seq, mini_float_vids_index, sparse_ft_miniv_train_path):
    samples = click_seq.map(lambda x: extract_seq_with_full_vid(x)) \
        .reduceByKey(lambda x, y: x + y) \
        .flatMap(lambda x: numerous_train_sample_format(x[0], x[1], mini_float_vids_index)) \
        .persist()

    delete_if_exist(sparse_ft_miniv_train_path)

    # extract train/test for sparse format
    samples.repartition(200).saveAsTextFile(sparse_ft_miniv_train_path)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("QB_VAE_RECALL").enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    ori_data = sc.textFile(input_path).map(lambda x: x.split("|")).map(lambda x: (x[0], x[1][:-3], x[4])).filter(
        lambda x: len(x[1]) == 10)

    # print("ori_cnt:{}".format(ori_data.count()))

    click_seq = ori_data.map(extract_seq)

    mini_float_vids_index = process_vid_sample(click_seq)

    # print("ori_data:{}".format(ori_data.take(2)))

    mini_float_vids_index = sc.broadcast(mini_float_vids_index)

    process_train_sample(click_seq, mini_float_vids_index,
                         "mdfs://cloudhdfs/mttsparknew/data/video/trs/float_vae/train_sample/train_sparse_ft_miniv/%s" % numerous_date_format)

