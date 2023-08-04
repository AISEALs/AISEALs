import sys
import os
sys.path.append(os.getcwd())    # 在命令行中，把当前路径加入到sys.path中
import argparse
import csv
import random
import traceback
import numpy as np
import pprint
import tensorflow as tf
from text_classification.data_processor.processor_manager import get_processor

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def _read_txt(input_file):
    with open(input_file, 'r+', encoding='UTF-8') as f:
        return f.readlines()


## 1. 把原始数据（未处理过的）转换成 csv文件（分类别）
def etl_rawdata2catecsv(processor, type='txt'):
    for fileName in os.listdir(processor.raw_data_dir):
        try:
            if FLAGS.debug_mode:
                if fileName != "tmp.txt":
                    continue
            if not fileName.endswith(type):
                continue

            filePath = os.path.join(processor.raw_data_dir, fileName)
            if type == 'txt':
                lines = _read_txt(filePath)
            else:
                lines = _read_tsv(filePath)

            for line in lines:
                try:
                    example = processor.convert_line2example(line)
                    if example:
                        processor.file_handlers.write_example(example.label, example.serialize_for_tsv())
                except Exception:
                    print("line:{}".format(line))
        except Exception:
            traceback.print_exc()
            continue
    processor.save_label_classes()
    processor.clear()

def convert_examples2tfrecord(processor, mode, output_shards=5, output_dir=None):
    def _pick_output_shard():
        return random.randint(0, output_shards - 1)

    if not output_dir:
        output_dir = processor.tfrecord_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_lens = []
    labels = processor.get_labels()
    for label in labels:
        info = processor.file_handlers.file_infos[label]
        num = info.train_num if mode == 'train' else info.eval_num
        data_lens.append(num)
        if mode == 'eval' and info.train_num:
            # Fast forward all files to skip the offset.
            count = 0
            while count < info.train_num:
                try:
                    count += 1
                    next(info.get_reader())
                except StopIteration:
                    continue

    writers = []

    for i in range(output_shards):
        path = os.path.join(output_dir, "{m}_{s}.tfrecord".format(m=mode, s=i))
        writers.append(tf.python_io.TFRecordWriter(path))

    indexes = range(len(labels))
    reading_order = list(map(int, np.concatenate([[i]*j for i,j in zip(list(indexes), data_lens)])))
    random.shuffle(reading_order)

    total_num = processor.get_train_num() if mode == 'train' else processor.get_eval_num()
    for (ex_index, c) in enumerate(reading_order):
        try:
            info = processor.file_handlers.file_infos[labels[c]]
            line = next(info.get_reader()).strip()
            if ex_index % 1000 == 0 or ex_index == total_num - 1:
                tf.logging.info("Writing example %d of %d" % (ex_index, total_num))
            example = processor.convert_example2tfexample(line, ex_index)
            if example == None:
                continue
            writers[_pick_output_shard()].write(example.SerializeToString())
        except StopIteration:
            continue

    for w in writers:
        w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--debug_mode",
        type="bool",
        default=False,
        # action='store_false',
        help="use debug mode")
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

    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter().pprint
    pp("FLAGS: " + str(FLAGS))
    pp("unparsed: " + str(unparsed))
    print(FLAGS.debug_mode)

    print("base_path:" + str(FLAGS.base_path))

    processor = get_processor(FLAGS.base_path, FLAGS.task_name, FLAGS.task_id, use_hdfs=False)

    etl_rawdata2catecsv(processor)
    processor.store_vocab()
    convert_examples2tfrecord(processor, mode='train')
    convert_examples2tfrecord(processor, mode='eval')
