# -*- coding:utf-8 -*-
'''
1.data_processor.py 把原始新闻转化为分词后以空格分割的文章。
2.vocabulary_processor.py 根据1生成结果，生成词库。
3.youliao_create_dataset.py 根据1和2的结果，生成tfrecord文件，供4使用。
4.youliao_helpers.py 读取3tfrecord取训练cnn models。
'''
'''
raw data: 158:/opt/youliao_syn/data 
grep "zjb-"  这样可以过滤出来
'''

import json
import traceback
import os
import csv
import random
import tensorflow as tf
from bert import tokenization
import numpy as np
import functools


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

    def serialize_for_tsv(self):
        return '\t'.join([self.guid, str(self.label), self.text])

class ExampleFileInfo(object):
    def __init__(self, writer, label):
        self.writer = writer
        # self.reader = reader
        self.total_num = 0
        self.train_num = 0
        self.eval_num = 0
        self.label = label

    def write(self, example):
        self.writer.write(example.serialize_for_tsv() + '\n')
        self.total_num += 1

    def calculate_and_filter(self):
        if self.total_num <= 10:
            return False
        self.eval_num = int(self.total_num * 0.1)
        if self.eval_num > 1024:
            self.eval_num = 1024
        self.train_num = self.total_num - self.eval_num
        return True

    def flush(self):
        self.writer.flush()
        self.writer.close()


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, input_dir):
        self.file_infos = {}
        self.labels = []
        self.input_dir = input_dir
        self.output_dir = os.path.join(input_dir, 'result')
        self.tfrecord_dir = os.path.join(self.input_dir, "tfrecord/")
        self.class_file = os.path.join(self.tfrecord_dir, "label.classes")
        self.total_num = 0
        self.train_num = 0
        self.eval_num = 0

    def pre_etl_examples(self):
        for fileName in os.listdir(self.input_dir):
            try:
                if fileName == 'result' or fileName == 'tfrecord':
                    continue
                filePath = os.path.join(self.input_dir, fileName)
                self._create_examples(self._read_txt(filePath))
            except Exception as e:
                traceback.print_exc()
                continue
        self.save_label_class()
        self.clear()


    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def load_classes(self, init_handle=False):
        with open(self.class_file, mode='r', encoding='UTF-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row['label']
                info = ExampleFileInfo(None, label)
                info.total_num = int(row['total_num'])
                info.train_num = int(row['train_num'])
                info.eval_num = int(row['eval_num'])
                if init_handle:
                    file_name = os.path.join(self.output_dir, '{r}.tsv'.format(r=label))
                    info.writer = open(file_name, 'r+', encoding='UTF-8')
                self.file_infos[label] = info
                self.labels.append(label)
                self.total_num += info.total_num
                self.train_num += info.train_num
                self.eval_num += info.eval_num
            tf.logging.info("load class file over")

    def clear(self):
        self.file_infos = {}
        self.labels = []
        self.total_num = 0
        self.train_num = 0
        self.eval_num = 0

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.labels

    def _create_examples(self, lines):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls, input_file):
        with open(input_file, 'r+', encoding='UTF-8') as f:
            return f.readlines()

    def write_file(self, example):
        label = example.label
        if label not in self.file_infos:
            output_file = os.path.join(self.output_dir, '{r}.tsv'.format(r=label))
            writer = open(output_file, 'w+', encoding='UTF-8')
            self.file_infos[label] = ExampleFileInfo(writer=writer, label=label)
        self.file_infos[label].write(example)

    def save_label_class(self):
        with tf.gfile.GFile(self.class_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "total_num", "train_num", "eval_num"])
            for (label, file_info) in self.file_infos.items():
                if file_info.calculate_and_filter():
                    writer.writerow([label, file_info.total_num, file_info.train_num, file_info.eval_num])
                    self.total_num += file_info.total_num
                    self.train_num += file_info.train_num
                    self.eval_num += file_info.eval_num
                file_info.flush()

    def convert_examples2tfrecord(self, mode, output_shards=5, output_dir=None, func=None):
        def _pick_output_shard():
            return random.randint(0, output_shards - 1)

        if not output_dir:
            output_dir = self.tfrecord_dir
        output_file = os.path.join(output_dir, "{s}.tfrecord".format(s=mode))

        data_lens = []
        for label in self.labels:
            info = self.file_infos[label]
            num = info.train_num if mode == 'train' else info.eval_num
            data_lens.append(num)
            # if mode == 'eval' and info.train_num:
            #     # Fast forward all files to skip the offset.
            #     count = 0
            #     while count < info.train_num:
            #         try:
            #             count += 1
            #             next(info.writer)
            #         except StopIteration:
            #             continue

        writers = []
        for i in range(output_shards):
            writers.append(
                tf.python_io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i,
                                                                 output_shards)))

        indexes = range(len(self.labels))
        reading_order = list(map(int, np.concatenate([[i]*j for i,j in zip(list(indexes), data_lens)])))
        random.shuffle(reading_order)

        total_num = self.train_num if mode == 'train' else self.eval_num
        for (ex_index, c) in enumerate(reading_order):
            try:
                info = self.file_infos[self.labels[c]]
                line = next(info.writer).strip()
                if ex_index % 1000 == 0 or ex_index == total_num - 1:
                    tf.logging.info("Writing example %d of %d" % (ex_index, total_num))
                example = self.parse_line_func(ex_index, line, func)
                writers[_pick_output_shard()].write(example.SerializeToString())
            except StopIteration:
                continue

        # Close all files
        for w in writers:
            w.close()

    def parse_line_func(self, ex_index, line, func=None):
        raise NotImplementedError()


class YouliaoProcessor(DataProcessor):
    """Processor for the youliao data set."""
    def __init__(self, input_dir):
        super(YouliaoProcessor, self).__init__(input_dir)
        self.num = 0

        self.stopwords=set()

    def get_train_examples(self):
        pass

    def _deal_line(self, line):
        try:
            class YouliaoRecord(object):
                pass
            line = line.split('zjb-')[1]
            decode_line = json.loads(line)
            content = decode_line['content']
            title = decode_line['title']
            x = ""
            if title:
                x += title
                x += " "
            if content:
                x += content
            if not x:
                return None
            x = data_helpers.clean_html(x)
            import jieba
            seg_text = jieba.cut(x.replace('\t',' ').replace('\n',' '))
            seg_text = filter(lambda x:x not in self.stopwords, seg_text)

            result = YouliaoRecord()
            result.x = ' '.join(seg_text)
            category = decode_line['category']
            if not category:
                return None
            category = category.split('-')[0]
            if category not in self.labels:
                self.labels.append(category)
            result.y = category
            return result
        except Exception as e:
            # traceback.print_exc()
            return None

    def load_stop_words(self):
        self.stopwords.add(' ')
        stopfilepath = 'stop_words_ch_utf8.txt'
        with open(stopfilepath, 'r+', encoding='UTF-8') as f:
            for l in f:
                self.stopwords.add(l.strip())

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        self.load_stop_words()
        for line in lines:
            result = self._deal_line(line)
            if not result:
                continue
            guid = "%s" % self.num
            self.num += 1
            text = tokenization.convert_to_unicode(result.x)
            label = tokenization.convert_to_unicode(result.y)
            example = InputExample(guid=guid, text=text, label=label)
            self.write_file(example)

    def parse_line_func(self, ex_index, line, func=None):
        label_list = self.get_labels()
        if not func:
            raise NotImplementedError
        return func(ex_index, line, label_list)


if __name__ == '__main__':
    processor = YouliaoProcessor("/Users/jiananliu/AISEALs/data/text_classification/tmp_youliao_raw_data")
    # processor.pre_etl_examples()
    # obj.get_labels()
    # obj.convert_examples_to_features('train')
    train_pattern = os.path.join(processor.tfrecord_dir, "eval.tfrecord*")
    # train_pattern = os.path.join(processor.tfrecord_dir, "train.tfrecord*")

    from run_classifier import file_based_input_fn_builder
    train_input_fn = file_based_input_fn_builder(
        tfrecord_pattern=train_pattern,
        seq_length=128,
        is_training=True,
        drop_remainder=True)

    dataset = train_input_fn(params={"batch_size": 1})
    with tf.Session() as sess:
        # sess.run(iterator.initializer)
        while True:
            try:
                x = sess.run([dataset])
                example = x[0]
                print(x)
                tf.logging.info("*** Example ***")
                # tf.logging.info("guid: %s" % (example.guid))
                # tf.logging.info("tokens: %s" % " ".join(
                #     [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in example["input_ids"]]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in example["input_mask"]]))
                tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in example["segment_ids"]]))
                tf.logging.info("label: (id = %d)" % example["label_ids"][0])
                break
            except tf.errors.OutOfRangeError:
                break
