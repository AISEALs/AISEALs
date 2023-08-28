#!/usr/bin/python
# coding=utf-8
import numerous

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tensorflow.keras as tf_keras
import math
from numerous.framework.dense_parameter import DenseParameter
from numerous.distributions.uniform import Uniform
from numerous.optimizers.movavg import Movavg
from numerous.optimizers.adam import Adam
from numerous.client import context
from numerous.optimizers.ftrl import FTRL


class DenoiseNet(object):
    def __init__(self):
        self.model_slice = "embedding"
        self.save_type = "int8_float32"
        self.combine_method = "sum"

    def get_denoise_net(self, input_layer, flush_num, is_training, name):
        ## weightnet dnn
        weightnet_embeding_merge = tf.stop_gradient(input_layer)

        item_out = []

        for i in range(flush_num):
            weight_layer1 = tf.layers.dense(weightnet_embeding_merge, units=512, activation=tf.nn.leaky_relu,
                                            name="weight_net1_{}_{}".format(name, i))
            weight_layer1 = tf.layers.batch_normalization(weight_layer1, training=is_training,
                                                          name="weight_net_bn1_{}_{}".format(name, i))
            weight_layer2 = tf.layers.dense(weight_layer1, units=512, activation=tf.nn.leaky_relu,
                                            name="weight_net2_{}_{}".format(name, i))
            weight_layer2 = tf.layers.batch_normalization(weight_layer2, training=is_training,
                                                          name="weight_net_bn2_{}_{}".format(name, i))
            weight_layer3 = tf.layers.dense(weight_layer2, units=32, activation=tf.nn.leaky_relu,
                                            name="weight_net3_{}_{}".format(name, i))
            weightnet_deep_out = tf.reduce_sum(weight_layer3, axis=1, keepdims=True)
            item_out.append(weightnet_deep_out)

        return tf.concat(item_out, axis=1, name="denoise_net_out")
