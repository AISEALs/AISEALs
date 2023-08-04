# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Creates training and eval data from Quickdraw NDJSON files.

This tool reads the NDJSON files from https://quickdraw.withgoogle.com/data
and converts them into tensorflow.Example stored in TFRecord files.

The tensorflow example will contain 3 features:
 shape - contains the shape of the sequence [length, dim] where dim=3.
 class_index - the class index of the class for the example.
 ink - a length * dim vector of the ink.

It creates disjoint training and evaluation sets.

python create_dataset.py \
  --ndjson_path ${HOME}/ndjson \
  --output_path ${HOME}/tfrecord

data_processor.py 把原始新闻转化为分词后以空格分割的文章。
vocabulary_processor.py 根据1生成结果，生成词库。
youliao_create_dataset.py 根据1和2的结果，生成tfrecord文件，供4使用。
youliao_helpers.py 读取3tfrecord取训练cnn models。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import random
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn


def convert_data(trainingdata_dir,
                 output_file,
                 classnames,
                 output_shards=10,
                 model='train',
                 vocab = None):
  """Convert training data from ndjson files into tf.Example in tf.Record.

  Args:
   trainingdata_dir: path to the directory containin the training data.
     The training data is stored in that directory as ndjson files.
   observations_per_class: the number of items to load per class.
   output_file: path where to write the output.
   classnames: array with classnames - is auto created if not passed in.
   output_shards: the number of shards to write the output in.
   offset: the number of items to skip at the beginning of each file.

  Returns:
    classnames: the class names as strings. classnames[classes[i]] is the
      textual representation of the class of the i-th data point.
  """

  def _pick_output_shard():
    return random.randint(0, output_shards - 1)

  file_handles = []
  data_lens = []
  # Open all input files.
  for filename in sorted(tf.gfile.ListDirectory(trainingdata_dir)):
    if filename.endswith(".ndjson"):
        file_type = "ndjson"
    elif filename.endswith(".csv") or filename.endswith(".tsv"):
        file_type = "csv"
    elif filename.endswith(".txt"):
        file_type = "txt"
    else:
        file_type = None

    if file_type == None:
      print("Skipping", filename)
      continue

    f = tf.gfile.GFile(os.path.join(trainingdata_dir, filename), "r")
    if file_type == "csv" or file_type == "tsv":
        file_handles.append(csv.DictReader(f, fieldnames=['guid', 'y', 'x'], delimiter='\t'))
    elif file_type == "txt":
        file_handles.append(f)
    class_name = filename.replace(file_type, "")
    if class_name not in classnames:
        classnames.append(class_name)

    total_size = len(tf.gfile.GFile(os.path.join(trainingdata_dir, filename), "r").readlines())
    eval_size = int(total_size * 0.1)
    if eval_size > 1024:
        eval_size = 1024
    train_size = total_size - eval_size
    if model == 'train':
        data_lens.append(train_size)
    else:
        data_lens.append(eval_size)

    if model == 'eval' and train_size:  # Fast forward all files to skip the offset.
      count = 0
      while count < train_size:
          try:
            count += 1
            next(file_handles[-1])
          except StopIteration:
              continue

  writers = []
  for i in range(FLAGS.output_shards):
    writers.append(
        tf.python_io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i,
                                                         output_shards)))

  # reading_order = list(range(len(file_handles))) * observations_per_class
  indexes = range(len(file_handles))
  reading_order = list(map(int, np.concatenate([[i]*j for i,j in zip(list(indexes), data_lens)])))

  random.shuffle(reading_order)

  for c in reading_order:
    try:
        line = next(file_handles[c])
        ink = None
        class_index = c
        while ink is None:
          ink = parse_line(line, vocab)
          if ink is None:
            print ("Couldn't parse ink from '" + line + "'.")
        features = {}
        features["label"] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[class_index]))
        #这里必须把np.int64 转成int，不然后面反序列化会出错
        tmp = [int(i) for i in ink]
        features["feature"] = tf.train.Feature(int64_list=tf.train.Int64List(value=tmp))
        f = tf.train.Features(feature=features)
        example = tf.train.Example(features=f)
        writers[_pick_output_shard()].write(example.SerializeToString())
    except StopIteration:
        continue


  # Close all files
  for w in writers:
    w.close()
  # for f in file_handles:
  #   f.close()
  # Write the class list.
  with tf.gfile.GFile(output_file + ".classes", "w") as f:
    for class_name in classnames:
      f.write(class_name + "\n")
  return classnames

def parse_line(sample, vocab):
    # 文本转为词ID序列，未知或填充用的词ID为0
    id_documents = list(vocab.transform([sample['x']]))
    return id_documents[0]


def main(argv):
  del argv
  vocab = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_dir)
  classnames = convert_data(
      FLAGS.ndjson_path,
      os.path.join(FLAGS.output_path, "training.tfrecord"),
      classnames=[],
      output_shards=FLAGS.output_shards,
      model='train',
      vocab = vocab)
  convert_data(
      FLAGS.ndjson_path,
      os.path.join(FLAGS.output_path, "eval.tfrecord"),
      classnames=classnames,
      output_shards=FLAGS.output_shards,
      model='eval',
      vocab = vocab)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--ndjson_path",
      type=str,
      default="/Users/jiananliu/work/python/AISEALs/data/text_classification/youliao_raw_data/result",
      help="Directory where the ndjson files are stored.")
  parser.add_argument(
      "--output_path",
      type=str,
      default="/Users/jiananliu/work/python/AISEALs/data/text_classification/youliao_raw_data/tfrecord_v1",
      help="Directory where to store the output TFRecord files.")
  parser.add_argument(
      "--output_shards",
      type=int,
      default=5,
      help="Number of shards for the output.")
  parser.add_argument(
      "--vocab_dir",
      type=str,
      default="/Users/jiananliu/AISEALs/text_classification/vocab.pickle",
      help="Directory where the ndjson files are stored.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
