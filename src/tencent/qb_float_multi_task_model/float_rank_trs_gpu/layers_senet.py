# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as K
from model_zoo.layers import MMoE
from model_zoo.layers import MLP

from typing import List


class MultiHeadSelfAttention(K.layers.Layer):
    """An implementation of the Multi-head self-attention module.
    """

    def __init__(self, prefix, attention_embedding_size, num_heads, num_blocks, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self._att_t = attention_embedding_size
        self._att_head = num_heads
        self._att_stack = num_blocks
        self.dropout_rate = dropout_rate
        self.wq, self.wk, self.wv = [], [], []
        self.dropout, self.layer_norm = [], []

    def build(self, input_shape):
        m = int(input_shape[1])  # slot num
        k = int(input_shape[2])  # feature embedding size
        t = self._att_t
        self.k = k
        self.m = m
        self.t = t
        self.h = self._att_head
        for j in range(self._att_stack):
            self.wq.append([])
            self.wk.append([])
            self.wv.append([])
            for i in range(self._att_head):
                self.wq[j].append(self.add_weight(name=self.prefix + '_' + str(j) + '_wq_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.1)))
                self.wk[j].append(self.add_weight(name=self.prefix + '_' + str(j) + '_wk_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.1)))
                self.wv[j].append(self.add_weight(name=self.prefix + '_' + str(j) + '_wv_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.1)))
            self.dropout.append(tf.keras.layers.Dropout(self.dropout_rate, name='dropout_{}'.format(j)))
            self.layer_norm.append(tf.keras.layers.LayerNormalization(axis=-1, name='layer_norm_{}'.format(j)))

    def mask(self, inputs, queries=None, keys=None):

        padding_num = -2 ** 32 + 1

        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)

        return outputs

    def mhsar(self, x, j, training):
        concat_input = []
        for i in range(self.h):
            # x (batch, m, k); wq (k, t)
            q = tf.tensordot(x, self.wq[j][i], axes=[2, 0])  # (batch, m, t)
            _k = tf.tensordot(x, self.wk[j][i], axes=[2, 0])  # (batch, m, t)
            v = tf.tensordot(x, self.wv[j][i], axes=[2, 0])  # (batch, m, t)
            z1 = tf.matmul(q, _k, transpose_b=True)  # (batch, m, m)
            z2 = z1 * (1 / tf.math.sqrt(tf.cast(self.k, tf.float32)))  # (batch, m, m)
            z2 = self.mask(z2, q, _k)
            z3 = tf.nn.softmax(z2, axis=2)  # (batch, m, m)
            z4 = tf.matmul(z3, v)  # (batch, m, t)
            z5 = tf.reshape(z4, shape=[-1, self.m * self.t])  # (batch, m * t)
            concat_input.append(z5)
        mhsa = tf.concat(concat_input, axis=1)  # (batch, m * t * h)
        x_r2 = tf.reshape(x, shape=[-1, self.m * self.k])  # (batch, m * k)
        mhsa = self.dropout[j](mhsa, training)
        z1 = mhsa + x_r2  # (batch, m * t * h)
        z1_norm = self.layer_norm[j](z1)
        z_r2 = z1_norm  # (batch, m * t * h)
        z_r3 = tf.reshape(z_r2, shape=[-1, self.m, self.t * self.h])  # (batch, m, t * h)
        return z_r2, z_r3

    def call(self, inputs, training):
        x = inputs
        concat_input = []
        for j in range(self._att_stack):
            z_r2, z_r3 = self.mhsar(x, j, training)
            concat_input.append(z_r2)
            x = z_r3
        affine_input = tf.concat(concat_input, axis=1)
        return affine_input, x


class MMoEModel(MMoE):
    def __init__(self,
                 task_names: List[str],
                 expert_num: int,
                 expert_hidden_sizes: List[int],
                 tower_hidden_sizes: List[int],
                 **kwargs):
        super(MMoEModel, self).__init__(task_names, expert_num, expert_hidden_sizes, tower_hidden_sizes, **kwargs)

    def build_expert(self, name: str):
        mlp = MLP(
            name=name,
            hidden_units=self._expert_hidden_sizes,
            use_bn=False,
            activation='selu',
            output_activation='selu')
        return mlp

    def build_tower(self, name: str):
        mlp = MLP(
            name=name,
            hidden_units=self._tower_hidden_sizes,
            activation='relu',
            use_bn=True,
            output_activation='relu')
        return mlp


class DenoiseLayer(K.layers.Layer):
    def __init__(self, **kwargs):
        super(DenoiseLayer, self).__init__(**kwargs)
        weightnet_dense_normal_init = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
        self.weightnet_dense1 = tf.keras.layers.Dense(512, kernel_initializer=weightnet_dense_normal_init,
                                                      name="weightnet_w1_{}".format(self._name))
        self.weightnet_bn1 = tf.keras.layers.BatchNormalization(name="weightnet_bn1_{}".format(self._name))
        self.weightnet_dense2 = tf.keras.layers.Dense(512, kernel_initializer=weightnet_dense_normal_init,
                                                      name="weightnet_w2_{}".format(self._name))
        self.weightnet_bn2 = tf.keras.layers.BatchNormalization(name="weightnet_bn2_{}".format(self._name))
        self.weightnet_dense3 = tf.keras.layers.Dense(32, kernel_initializer=weightnet_dense_normal_init,
                                                      name="weightnet_w3_{}".format(self._name))

        self.output_dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training):
        embedding, weightnet_lr_sum = inputs
        weightnet_embeding_merge = tf.stop_gradient(embedding)
        weightnet_layer_1 = self.weightnet_dense1(weightnet_embeding_merge)
        weightnet_layer_1 = tf.nn.leaky_relu(weightnet_layer_1, alpha=0.02)
        weightnet_layer_1 = self.weightnet_bn1(weightnet_layer_1, training=training)

        weightnet_layer_2 = self.weightnet_dense2(weightnet_layer_1)
        weightnet_layer_2 = tf.nn.leaky_relu(weightnet_layer_2, alpha=0.02)
        weightnet_layer_2 = self.weightnet_bn2(weightnet_layer_2, training=training)

        weightnet_layer_3 = self.weightnet_dense3(weightnet_layer_2)
        weightnet_layer_3 = tf.nn.leaky_relu(weightnet_layer_3, alpha=0.02)

        weightnet_lr_deep_in = tf.concat([weightnet_lr_sum, weightnet_layer_3], axis=1)
        weightnet_logit = self.output_dense(weightnet_lr_deep_in)

        return weightnet_logit


class FSSENet(tf.keras.layers.Layer):
    def __init__(self, name, num_units, fea_len, **kwargs):
        self.num_units = num_units
        self.fea_len = fea_len
        super(FSSENet, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        print("Building shape: {}".format(input_shape))
        self.att_pooling = tf.keras.layers.Dense(1, use_bias=False, name=self.name + "_att_pooling")  # None * F * K
        self.dense_pooling = tf.keras.layers.Dense(1, use_bias=False, name=self.name + "_dense_pooling")  # None * F * K
        self.layer_se = tf.keras.layers.Dense(
            self.num_units,
            activation=tf.nn.relu,
            name=self.name +
            "_squeeze")  # None * d
        self.layer_top = tf.keras.layers.Dense(
            self.fea_len,
            activation=tf.nn.sigmoid,
            name=self.name +
            "_excitation")  # None * d
        super(FSSENet, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, attention_inputs, dense_inputs):
        attention_inputs = tf.squeeze(self.att_pooling(attention_inputs), axis=-1)  # None * F
        dense_inputs = tf.squeeze(self.dense_pooling(dense_inputs), axis=-1)  # None * F
        emb_pool = tf.concat([attention_inputs, dense_inputs], axis=1)
        gate_weight = self.layer_se(emb_pool)  # None * d
        gate_weight = self.layer_top(gate_weight)  # None * F
        gate_weight = tf.expand_dims(gate_weight, 2)  # None * F * 1
        return gate_weight
