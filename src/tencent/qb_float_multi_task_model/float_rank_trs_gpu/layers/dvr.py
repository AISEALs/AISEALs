# -*- coding: utf-8 -*-
# Copyright 2023 Tencent Inc.  All rights reserved.
# author: waferzhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal


class WatchTimeGainUpdater(K.layers.Layer):
    def __init__(self, bin_boundaries, **kwargs):
        super(WatchTimeGainUpdater, self).__init__(**kwargs)
        self.bin_boundaries = bin_boundaries
        self.num_bins = len(bin_boundaries) + 1
        self.mean_time = self.add_weight(name='{}_mean_time'.format(self._name),
                                         shape=[self.num_bins],
                                         initializer=K.initializers.Constant(0.0),
                                         trainable=False)
        self.variance_time = self.add_weight(name='{}_variance_time'.format(self._name),
                                             shape=[self.num_bins],
                                             initializer=K.initializers.Constant(0.0),
                                             trainable=False)
        self.count = self.add_weight(name='{}_count'.format(self._name),
                                     shape=[self.num_bins],
                                     initializer=K.initializers.Constant(0.0),
                                     trainable=False)

    def call(self, inputs, **kwargs):
        watch_times, durations = inputs
        watch_times = tf.reshape(watch_times, [-1])
        durations = tf.reshape(durations, [-1])
        bins = tf.raw_ops.Bucketize(input=durations, boundaries=self.bin_boundaries)
        mean_time_updates = []
        variance_time_updates = []
        count_updates = []

        for i in range(self.num_bins):
            mean_time = self.mean_time[i]
            variance_time = self.variance_time[i]
            count = self.count[i]
            condition = tf.math.equal(bins, tf.cast(i, tf.int32))
            bin_counts = tf.where(condition, 1.0, 0.0)

            # update count
            count_update = count + tf.reduce_sum(bin_counts)
            count_updates.append(count_update)

            # update variance_time
            bin_variance_times = tf.where(condition,
                                          (count_update - 1.0) / (tf.math.square(count_update) + 1e-9) * tf.math.square(watch_times - mean_time) - 1.0 / (count_update + 1e-9) * variance_time,
                                          0.0)
            variance_time_update = variance_time + tf.reduce_sum(bin_variance_times)
            variance_time_updates.append(variance_time_update)

            # update mean_time
            bin_mean_times = tf.where(condition, (watch_times - mean_time) / (count_update + 1e-9), 0.0)
            mean_time_update = mean_time + tf.reduce_sum(bin_mean_times)
            mean_time_updates.append(mean_time_update)

        new_count = tf.stack(count_updates, axis=0)
        new_variance_time = tf.stack(variance_time_updates, axis=0)
        new_mean_time = tf.stack(mean_time_updates, axis=0)
        new_std_time = tf.math.sqrt(new_variance_time)

        batch_mean_times = tf.gather(new_mean_time, bins)
        batch_std_times = tf.gather(new_std_time, bins)
        batch_watch_time_gains = tf.reshape((watch_times - batch_mean_times) / (batch_std_times + 1e-9), [-1, 1])

        self.count.assign(new_count)
        self.variance_time.assign(new_variance_time)
        self.mean_time.assign(new_mean_time)

        return batch_watch_time_gains


def DVR(output_independent):
    output = PredictionLayer(task='regression', name='prediction')(output_independent)

    independent_predictor = GradientReversalLayer()(output_independent)
    reconstructor = K.layers.Dense(1, name='reconstructor')
    independent_reconstruct = reconstructor(independent_predictor)
    independent_reconstruct = PredictionLayer(task='regression', name='independent_reconstruct')(
        independent_reconstruct)
    return [output, independent_reconstruct]


@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return -1 * dy

    return x, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs):
        return GradientReversalOperator(inputs)


class PredictionLayer(K.layers.Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
