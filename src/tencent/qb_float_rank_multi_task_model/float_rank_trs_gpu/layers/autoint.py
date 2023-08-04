# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as K


class AutoIntModel(K.layers.Layer):
    def __init__(self, prefix, attention_embedding_size, num_heads, num_blocks, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self._att_t = attention_embedding_size
        self._att_head = num_heads
        self._att_stack = num_blocks
        self.dropout_rate = dropout_rate
        self.wq, self.wk, self.wv = {}, {}, {}
        self.dropout, self.layer_norm = [], []

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

    def build(self, input_shape):
        m = int(input_shape[1])  # slot num
        k = int(input_shape[2])  # feature embedding size
        t = self._att_t
        self.k = k
        self.m = m
        self.t = t
        self.h = self._att_head
        for i in range(self._att_stack):
            prefix = self.prefix + "_" + str(i)
            for j in range(self._att_head):
                self.wq[prefix + '_wq_' + str(j)] = self.add_weight(name=prefix + '_wq_' + str(j),
                                                                    shape=[k, t],
                                                                    dtype=tf.float32,
                                                                    initializer=K.initializers.RandomNormal(mean=0.0,
                                                                                                            stddev=0.1))
                self.wk[prefix + '_wk_' + str(j)] = self.add_weight(name=prefix + '_wk_' + str(j),
                                                                    shape=[k, t],
                                                                    dtype=tf.float32,
                                                                    initializer=K.initializers.RandomNormal(mean=0.0,
                                                                                                            stddev=0.1))
                self.wv[prefix + '_wv_' + str(j)] = self.add_weight(name=prefix + '_wv_' + str(j),
                                                                    shape=[k, t],
                                                                    dtype=tf.float32,
                                                                    initializer=K.initializers.RandomNormal(mean=0.0,
                                                                                                            stddev=0.1))
            self.dropout.append(tf.keras.layers.Dropout(self.dropout_rate, name='dropout_{}'.format(i)))
            self.layer_norm.append(tf.keras.layers.LayerNormalization(axis=-1, name='layer_norm_{}'.format(i)))

    def mhsar(self, x, i, training):
        concat_input = []
        for j in range(self.h):
            # x (batch, m, k); wq (k, t)
            q = tf.tensordot(x, self.wq[self.prefix + '_' + str(i) + '_wq_' + str(j)], axes=[2, 0])  # (batch, m, t)
            _k = tf.tensordot(x, self.wk[self.prefix + '_' + str(i) + '_wk_' + str(j)], axes=[2, 0])  # (batch, m, t)
            v = tf.tensordot(x, self.wv[self.prefix + '_' + str(i) + '_wv_' + str(j)], axes=[2, 0])  # (batch, m, t)
            z1 = tf.matmul(q, _k, transpose_b=True)  # (batch, m, m)
            z2 = z1 * (1 / tf.math.sqrt(tf.cast(self.k, tf.float32)))  # (batch, m, m)
            z2 = self.mask(z2, q, _k)
            z3 = tf.nn.softmax(z2, axis=2)  # (batch, m, m)
            z4 = tf.matmul(z3, v)  # (batch, m, t)
            z5 = tf.reshape(z4, shape=[-1, self.m * self.t])  # (batch, m * t)
            concat_input.append(z5)
        mhsa = tf.concat(concat_input, axis=1)  # (batch, m * t * h)
        x_r2 = tf.reshape(x, shape=[-1, self.m * self.k])  # (batch, m * k)
        mhsa = self.dropout[i](mhsa, training)
        z1 = mhsa + x_r2  # (batch, m * t * h)
        z1_norm = self.layer_norm[i](z1)
        z_r2 = z1_norm  # (batch, m * t * h)
        z_r3 = tf.reshape(z_r2, shape=[-1, self.m, self.t * self.h])  # (batch, m, t * h)
        return z_r2, z_r3

    def call(self, inputs, training):
        x = inputs
        concat_input = []
        for i in range(self._att_stack):
            z_r2, z_r3 = self.mhsar(x, i, training)
            concat_input.append(z_r2)
            x = z_r3
        affine_input = tf.concat(concat_input, axis=1)
        return affine_input
