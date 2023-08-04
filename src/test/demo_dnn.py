#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""The demo of dnn model implementation """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numerous

import tensorflow as tf
import numpy as np

from numerous.optimizers.optimizer_context import OptimizerContext
from numerous.framework.dense_parameter import DenseParameter
from numerous.optimizers.adam import Adam
from numerous.optimizers.sgd import SGD
from numerous.optimizers.ftrl import FTRL
from numerous.optimizers.adagrad import Adagrad
from numerous.optimizers.amsgrad import Amsgrad
from numerous.optimizers.adadelta import Adadelta
from numerous.distributions.uniform import Uniform
from numerous.distributions.normal import Normal


# task
numerous.training.Task(model_name = "dnn_test", worker_thread = 4, worker_async_thread = 2, server_thread = 2)

# saver
numerous.training.Saver(dump_interval = "dir:1", always_complete = 1)

# global optimizer
OptimizerContext().set_optimizer(optimizer=Adam(rho1 = 0.9, rho2 = 0.999, eps = 0.001, time_thresh = 5000))

# label
y_data = tf.placeholder(tf.float32)
numerous.reader.LabelPlaceholder(y_data)

# embedding
embedding_sum_layers = {}
dimension = 6
dnn_slots = (601,602,603,616,734,735,736,737,626,622,623,624,625,738,739,740,741,742,743,644)

for slot_id in dnn_slots:
    embedding_w, slots = numerous.framework.SparseEmbedding(
        embedding_dim=dimension,
        slot_ids=[str(slot_id)])
    embedding = tf.matmul(slots, embedding_w)
    embedding_sum_layers[slot_id] = embedding

embedding_layer = tf.concat(embedding_sum_layers.values(), axis=1)

# dnn weights
fc_layer_w1 = tf.get_variable("w1", shape=[len(dnn_slots) * dimension, 512], dtype=tf.float32)
fc_layer_w2 = tf.get_variable("w2", shape=[512, 256], dtype=tf.float32)
fc_layer_w3 = tf.get_variable("w3", shape=[256, 128], dtype=tf.float32)
fc_layer_w4 = tf.get_variable("w4", shape=[128, 128], dtype=tf.float32)
fc_layer_w5 = tf.get_variable("w5", shape=[128, 128], dtype=tf.float32)
fc_layer_w6 = tf.get_variable("w6", shape=[128, 1], dtype=tf.float32)

fc_layer_1 = tf.nn.relu(tf.matmul(embedding_layer, fc_layer_w1))
fc_layer_2 = tf.nn.relu(tf.matmul(fc_layer_1, fc_layer_w2))
fc_layer_3 = tf.nn.relu(tf.matmul(fc_layer_2, fc_layer_w3))
fc_layer_4 = tf.nn.relu(tf.matmul(fc_layer_3, fc_layer_w4))
fc_layer_5 = tf.nn.relu(tf.matmul(fc_layer_4, fc_layer_w5))
logits = tf.matmul(fc_layer_5, fc_layer_w6)

# predict value
y_pred   = tf.sigmoid(logits)

# loss
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, 
                                                              labels = y_data))

# start numerous
numerous.estimator.Estimator().legacy_run([y_pred], loss)

