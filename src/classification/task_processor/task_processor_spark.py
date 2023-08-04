import os
import random
import argparse
import pprint
import re
from pyspark.sql import SparkSession
from tools import tools
from text_classification.common_tools.shuffle_generator import shuffle_all_in_cache
from text_classification.protobuf.proto_tools import write_record
from text_classification.data_processor.data_processor import Example
from text_classification.data_processor.processor_manager import get_processor
'''
shell调试：
task_processor --py-files youliao_bert_processor.py,data_processor.py,tools.py

提交运行:
spark-submit --py-files youliao_bert_processor.py,data_processor.py,tools.py \
task_processor_spark.py <base_dir>
'''


def convert_rawdata2catedata(spark, processor):
    if FLAGS.debug_mode:
        raw_data_rdd = spark.sparkContext.textFile(processor.raw_data_dir + "/2018-11-11")
    else:
        raw_data_rdd = spark.sparkContext.textFile(processor.raw_data_dir + "/*.txt")
    examples = raw_data_rdd.repartition(FLAGS.partition_num).map(processor.convert_line2example).filter(lambda x: x is not None).cache()
    print("total examples count:" + str(examples.count()))

    labels = examples.map(lambda x: x.label).distinct().collect()

    cate_rdd = examples.map(lambda e: (labels.index(e.label), e)).\
        partitionBy(numPartitions=len(labels)).map(lambda p: p[1])

    # 做一次全局打乱(shuffle)
    cate_rdd = cate_rdd.mapPartitions(shuffle_all_in_cache).cache()

    #这里要求每个类别数量不超过 100w个
    def gen_guid_func(s, iterator):
        num = 0
        base = 1000000
        for example in iterator:
            example.guid = num + base*s
            example.cuid = num
            num += 1
            if num >= base:
                raise RuntimeError
            yield example.serialize_for_tsv()

    cate_rdd.mapPartitionsWithIndex(gen_guid_func).saveAsTextFile(processor.cate_data_dir)

    def gen_label_total_num_func(s, iterator):
        label = labels[s]
        total_num = 0
        for _ in iterator:
            total_num = total_num + 1
        yield (label, total_num)

    label_num = cate_rdd.mapPartitionsWithIndex(gen_label_total_num_func).collect()
    print("label num:" + str(label_num))
    for (label, num) in label_num:
        processor.update_cate_file_info(label, num)

    processor.save_label_classes()


def generate_vocab_file(spark, processor):
    cate_data_rdd = spark.sparkContext.textFile(processor.cate_data_dir)
    word2num_dict = cate_data_rdd.repartition(FLAGS.partition_num).\
        map(lambda s: Example.deserialize_from_str(s)).\
        filter(lambda x: x is not None).\
        flatMap(lambda e: re.split("\x001| ", e.line.strip())).\
        map(lambda x: (x, 1)).\
        countByKey()

    print("word -> num dict num:{} size:{}".format(len(word2num_dict), sum(word2num_dict.values())))
    from common_tools import VocabularyProcessor
    max_document_length = 512
    min_frequency = 2
    # 序列长度填充或截取到512，删除词频<=2的词
    vocab = VocabularyProcessor(max_document_length, min_frequency)
    def iter_documents():
        for (label, num) in word2num_dict.items():
            for _ in range(num):
                yield label
    vocab.fit(iter_documents())
    client = tools.get_hdfs_client()
    vocab_path = os.path.join(processor.task_dir, 'vocab.pickle')
    with client.write(vocab_path) as f:
        try:
            # pylint: disable=g-import-not-at-top
            import cPickle as pickle
        except ImportError:
            # pylint: disable=g-import-not-at-top
            import pickle
        f.write(pickle.dumps(vocab))
    print("vocab.pick save path: {}".format(vocab_path))


def convert_catedata2tfrecord(spark, processor):
    labels = processor.get_labels()

    cate_data_rdd = spark.sparkContext.textFile(processor.cate_data_dir)

    output_shards = 5

    # 0-4 train, 5-9 eval
    def _pick_output_shard(label, cuid):
        train_num = processor.get_file_handler(label).train_num
        eval_num = processor.get_file_handler(label).eval_num
        if cuid < train_num:
            return random.randint(0, output_shards - 1)
        elif cuid < train_num + eval_num:
            return random.randint(output_shards, output_shards*2-1)
        else:
            return -1

    labels_map = processor.labels_mapping()

    def trans_func(e):
        shard_id = _pick_output_shard(e.label, e.cuid)
        if len(labels_map) > 0:
            e.label = labels_map[e.label]
        tf_example_str = processor.convert_example2tfexample(e.serialize_for_tsv(), e.cuid).SerializeToString()
        return shard_id, tf_example_str

    # 预先把cate_rdd均匀成64份，均衡executors运行时间。
    rdd = cate_data_rdd.repartition(FLAGS.partition_num).\
        map(lambda s: Example.deserialize_from_str(s)).\
        filter(lambda e: e is not None and e.label in labels).\
        map(lambda e: trans_func(e)).\
        filter(lambda x: x[0] != -1).\
        partitionBy(numPartitions=output_shards*2)

    rdd = rdd.mapPartitions(shuffle_all_in_cache).cache()

    if FLAGS.use_hdfs:
        client = tools.get_hdfs_client()
        if not client.status(processor.tfrecord_dir, strict=False):
            client.makedirs(processor.tfrecord_dir)
    else:
        if not os.path.exists(processor.tfrecord_dir):
            os.makedirs(processor.tfrecord_dir)

    def f(s, iterator):
        mode = "train" if s < output_shards else "eval"
        path = os.path.join(processor.tfrecord_dir, "{m}_{s}.tfrecord".format(m=mode, s=(s % output_shards)))
        def write_iterator(f):
            num = 0
            for i in iterator:
                if not i:
                    print("NOTICE why i is None")
                else:
                    write_record(f, i)
                num += 1
            return num

        if FLAGS.use_hdfs:
            print("NOTICE use hdfs")
            client = tools.get_hdfs_client()
            with client.write(path) as f:
                num = write_iterator(f)
        else:
            with open(path, 'wb+') as f:
                num = write_iterator(f)
        yield (s, num)

    print(rdd.map(lambda p: p[1]).mapPartitionsWithIndex(f).collect())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--debug_mode",
        type="bool",
        default=True,
        help="use debug mode")
    parser.add_argument(
        "--use_hdfs",
        type=bool,
        default=True,
        help="use hdfs file")
    parser.add_argument(
        "--partition_num",
        type=int,
        default=64,
        help="the number of the repartition num")
    parser.add_argument(
        "--base_path",
        type=str,
        default="/Users/jiananliu/AISEALs/data/text_classification",
        help="refer to data_processor path tree")
    parser.add_argument(
        "--task_name",
        type=str,
        default="tribe_labels",
        help="task name")
    parser.add_argument(
        "--task_id",
        type=str,
        default="20190708",
        help="task id")
    parser.add_argument(
        "--multiple",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="是否是多分类问题")
    parser.add_argument(
        "--label_name",
        type=str,
        default="心情_情绪_想法表达",
        help="二分类时生效，其他类归为:其他")


    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter().pprint
    pp(FLAGS)
    pp(unparsed)

    base_path = FLAGS.base_path
    task_name = FLAGS.task_name
    task_id = FLAGS.task_id

    print("base_path:" + str(base_path))

    processor = get_processor(base_path, task_name, task_id, use_hdfs=FLAGS.use_hdfs, multiple=FLAGS.multiple, label_name=FLAGS.label_name)

    spark = SparkSession.builder.\
        appName("task_processor_{}_{}".format(task_name, task_id)).\
        getOrCreate()

    # convert_rawdata2catedata(spark, processor)
    generate_vocab_file(spark, processor)
    convert_catedata2tfrecord(spark, processor)

    spark.stop()
