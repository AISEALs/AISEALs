#!/usr/bin/python
# coding=utf-8
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tensorflow.keras as tf_keras
import math
from numerous.framework.dense_parameter import DenseParameter
from numerous.optimizers.movavg import Movavg
from numerous.optimizers.sgd import SGD
from numerous.optimizers.adam import Adam
from numerous.optimizers.ftrl import FTRL
from numerous.client import context


class Model(object):
    def __init__(self, input, attention_embedding_size, num_heads, num_blocks, keep_prob, name):
        self.input = input
        self._att_t = attention_embedding_size
        self._att_head = num_heads
        self._att_stack = num_blocks
        self.keep_prob = keep_prob
        self.name = name

    def mask(self, inputs, queries=None, keys=None):

        padding_num = -2 ** 32 + 1

        # Generate masks
        # masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        # masks = tf.expand_dims(masks, 1) # (N, 1, T_k)
        # masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)
        # inputs shape: [bz, 5, size_of_concat_emb]

        x_shape = inputs.get_shape().as_list()
        b = tf.shape(inputs)[0]  # batch size
        m = int(x_shape[1])  # slot num
        assert m % 5 == 0  # 序列长度为5
        item_num = int(m / 5)
        mask_list = []
        for i in range(1, 5 + 1):
            mask1 = [1 if x < item_num * i else 0 for x in range(m)]
            t1 = tf.convert_to_tensor(mask1)
            t1 = tf.tile(tf.reshape(t1, [1, m]), [item_num, 1])
            mask_list.append(t1)
        mask_one = tf.concat(mask_list, axis=0)     # 是一个下三角矩阵
        masks = tf.tile(tf.expand_dims(mask_one, axis=0), [b, 1, 1])
        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)

        return outputs

    def mhsar(self, x, prefix):
        """multi head self attention & resnet
        input x: (batch, m, k)
        output z_r2: (batch, m * t * h)
        output z_r3: (batch, m, t * h)
        """
        x_shape = x.get_shape().as_list()

        m = x_shape[1]  # slot num
        k = x_shape[2]  # feature embedding size
        t = self._att_t  # attention embedding size
        h = self._att_head  # head num
        m = int(m)
        k = int(k)
        t = int(t)
        h = int(h)

        # 支持不同的优化器和参数初始化
        Optimizer = Adam(rho1=0.9, rho2=0.999, eps=0.005, time_thresh=5000)
        dense_normal_init = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32)

        concat_input = []
        for i in range(h):
            wq = tf.get_variable(prefix + '_wq_' + str(i),
                                 shape=[k, t],
                                 dtype=tf.float32,
                                 initializer=dense_normal_init)
            DenseParameter(wq, optimizer=Optimizer)
            # self.add_param(wq)
            wk = tf.get_variable(prefix + '_wk_' + str(i),
                                 shape=[k, t],
                                 dtype=tf.float32,
                                 initializer=dense_normal_init)
            DenseParameter(wk, optimizer=Optimizer)
            # self.add_param(wk)
            wv = tf.get_variable(prefix + '_wv_' + str(i),
                                 shape=[k, t],
                                 dtype=tf.float32,
                                 initializer=dense_normal_init)
            DenseParameter(wv, optimizer=Optimizer)
            # self.add_param(wv)

            # x (batch, m, k); wq (k, t)
            q = tf.tensordot(x, wq, axes=[2, 0], name="q")  # (batch, m, t)
            _k = tf.tensordot(x, wk, axes=[2, 0], name="k")  # (batch, m, t)
            v = tf.tensordot(x, wv, axes=[2, 0], name="v")  # (batch, m, t)
            z1 = tf.matmul(q, _k, transpose_b=True)  # (batch, m, m)
            z2 = z1 * (1 / np.sqrt(k))  # (batch, m, m)
            # mask
            z2 = self.mask(z2, q, _k)
            z3 = tf.nn.softmax(z2, axis=2, name="attention_softmax")  # (batch, m, m)
            # print sofmax
            # z3 = tf.Print(z3, [z3,x], summarize=30, message="softmax ouput: ")
            z4 = tf.matmul(z3, v)  # (batch, m, t)
            z5 = tf.reshape(z4, shape=[-1, m * t])  # (batch, m * t)
            concat_input.append(z5)
        mhsa = tf.concat(concat_input, axis=1)  # (batch, m * t * h)

        # wres = tf.get_variable(prefix + '_wres_',
        #                       shape=[m * t * h, m * k],
        #                       dtype=tf.float32)
        # DenseParameter(wres, optimizer=Optimizer, distribution=normal)
        # self.add_param(wres)
        x_r2 = tf.reshape(x, shape=[-1, m * k])  # (batch, m * k)
        # dropout
        mhsa = tf.nn.dropout(mhsa, self.keep_prob)
        z1 = mhsa + x_r2  # (batch, m * t * h)
        z1_norm = tf_keras.layers.LayerNormalization(axis=-1)(z1)
        z_r2 = z1_norm  # (batch, m * t * h)
        z_r3 = tf.reshape(z_r2, shape=[-1, m, t * h])  # (batch, m, t * h)
        return z_r2, z_r3

    def run(self):
        x = self.input
        concat_input = []
        for i in range(self._att_stack):
            z_r2, z_r3 = self.mhsar(x, 'mhsar_' + str(i) + "_" + self.name)
            concat_input.append(z_r2)
            x = z_r3
        affine_input = tf.concat(concat_input, axis=1, name="attention_embedding")
        # self._z = self.affine_output(affine_input)
        return affine_input
