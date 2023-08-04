# import pyspark
# import random
import enum
from pyspark import SparkContext
import os
import shutil
import sys
import numpy as np


def format_label_feature(sample):
    sample = sample.split("|")
    labels = [0, 0, 0, 0]
    idxs = [9, 10, 11, 12]  # idx in ext_info: like, share, read, write
    time = 0
    ext_info = sample[11].split(";")
    if len(ext_info) < 38:
        return
    click = int(sample[7])

    readTime = np.uint64(ext_info[4].split(":")[1])  # idx in ext_info: readTime=4
    time += int(readTime & np.uint64(0xffffffff << 32)) >> 32
    time += (readTime & np.uint64(0xffffffff))

    for i, idx in enumerate(idxs):
        labels[i] += int(ext_info[idx].split(":")[1])


    labels.insert(0, time)
    labels.insert(0, click)
    # labels: click, time, like, share, read, write

    es_raw = ext_info[37]

    if es_raw != "^":
        output_sample = ["", "", ""]
        key = "{}_{}".format(sample[4], sample[5])
        
        items = []
        
        es_raw_list = es_raw.split("^")
        for es_key in es_raw_list:
            t = es_key.split(":")
            if len(t) != 2:
                continue
            t2 = t[1].split("#")
            for item in t2:
                t3 = item.split(",")
                if len(t3) != 3:
                    continue
                items.append(t[0] + "_" + item)
        
        items_str = "\t".join(items)
        
        output_sample[0] = key
        output_sample[1] = labels
        output_sample[2] = items_str
        return key, output_sample
    else: 
        return
    

def reduce_sample_pair(x1, x2):
    labels1 = x1[1]
    for i in range(len(labels1)):
        labels1[i] += x2[1][i]
    return x1

def filter_and_tostring(sample):
    sample = sample[1]
    label = "_".join([str(x) for x in sample[1]])
    sample[1] = label
    out = "\t".join(sample)
    return out


if __name__ == "__main__":
    local = int(sys.argv[1])
    ds = sys.argv[2]
    future = int(sys.argv[3])
    sc = SparkContext()

    if local:
        rdd = sc.textFile("../mtl/samples/part-00000")
    else:
        rdd = sc.textFile("hdfs://qy-pcg-1-v2/user/tdw/warehouse/mttsparknew_mvideo.db/t_sh_atta_v1_0b500017091/ds=" + ds)

    rdd = rdd.map(format_label_feature)
    rdd = rdd.filter(lambda x: x is not None)
    rdd = rdd.reduceByKey(reduce_sample_pair)
    rdd = rdd.map(filter_and_tostring)
    rdd = rdd.filter(lambda x: x is not None).repartition(1)


    ds = "{}-{}-{}/{}".format(ds[0:4], ds[4:6], ds[6:8], ds[8:10])
    rdd.saveAsTextFile("mdfs://cloudhdfs/mttsparknew/user/mtt/mvideo/carliu/es_sample/{}".format(ds))

