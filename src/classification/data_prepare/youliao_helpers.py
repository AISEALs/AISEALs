'''
data_processor.py 把原始新闻转化为分词后以空格分割的文章。
vocabulary_processor.py 根据1生成结果，生成词库。
youliao_create_dataset.py 根据1和2的结果，生成tfrecord文件，供4使用。
youliao_helpers.py 读取3tfrecord取训练cnn models。
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf


def get_input_fn(mode, tfrecord_pattern, batch_size):
    """Creates an input_fn that stores all the data in memory.

    Args:
     mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
     tfrecord_pattern: path to a TF record file created using create_dataset.py.
     batch_size: the batch size to output.

    Returns:
      A valid input_fn for the models estimator.
    """

    def _parse_tfexample_fn(example_proto, mode):
        """Parse a single record which is expected to be a tensorflow.Example."""
        feature_to_type ={
            'feature': tf.FixedLenFeature([512], dtype=tf.int64)
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            # The labels won't be available at inference time, so don't add them
            # to the list of feature_columns to be read.
            feature_to_type["label"] = tf.FixedLenFeature([1], dtype=tf.int64)

        parsed_features = tf.parse_single_example(example_proto, feature_to_type)

        labels = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = parsed_features['label']
        # parsed_features["feature"] = tf.sparse_tensor_to_dense(parsed_features["feature"])
        return parsed_features['feature'], labels

    def _input_fn():
        """Estimator `input_fn`.

        Returns:
          A tuple of:
          - Dictionary of string feature name to `Tensor`.
          - `Tensor` of target labels.
        """
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
        # Preprocesses 10 files concurrently and interleaves records from each file.
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        buffer_size = batch_size * 3 + 100
        dataset.prefetch(buffer_size)
        # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size))
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=functools.partial(_parse_tfexample_fn, mode=mode), batch_size=batch_size, num_parallel_calls=5))
        dataset = dataset.prefetch(5)
        # features, labels = dataset.make_one_shot_iterator().get_next()
        # return features, labels
        # dataset = dataset.make_one_shot_iterator().get_next()
        return dataset

    def _input_fn2():
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)

        # not one data source parallel get data
        # dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        #     tf.data.TFRecordDataset, cycle_length=10))

        buffer_size = batch_size * 3 + 100
        # dataset.prefetch(buffer_size)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size,count=6))
        # dataset = dataset.shuffle(buffer_size=buffer_size)
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #     map_func=functools.partial(_parse_tfexample_fn, mode=mode), batch_size=batch_size, num_parallel_calls=5))
        # Preprocesses 10 files concurrently and interleaves records from each file.
        # dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda x:_parse_tfexample_fn(x, mode=mode), num_parallel_calls=5)
        # dataset = dataset.batch(batch_size)
        dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
        dataset = dataset.prefetch(3)
        return dataset

    return _input_fn2

if __name__ == '__main__':
    pattern = "/Users/jiananliu/work/python/AISEALs/data/text_classification/tfrecord/training.tfrecord-*"
    dataset = get_input_fn("train", pattern, 8)()

    with tf.Session() as sess:
        # sess.run(iterator.initializer)
        while True:
            try:
                x = sess.run([dataset])
                print(x[0][1].reshape(1, 8))
            except tf.errors.OutOfRangeError:
                break


