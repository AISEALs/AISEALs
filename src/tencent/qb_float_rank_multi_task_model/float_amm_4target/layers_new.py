# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as K

from model_zoo.layers.base_layer import BaseLayer
from typing import List


class MultiHeadSelfAttention(K.layers.Layer):
    """A implementation of the Multi-head self-attention module.

    """

    def __init__(self, attention_embedding_size, num_heads, num_blocks, **kwargs):
        """Create a transformer, which is a wrapper of the MultiHeadAttention.

        Args:
            layer_num (int): The number of MultiHeadAttention layers.
            head_num (int): The number of attention heads.
            hidden_size (int): The dimension of the input.
        """
        super().__init__(**kwargs)

        self._att_t = attention_embedding_size
        self._att_head = num_heads
        self._att_stack = num_blocks
        self.wq, self.wk, self.wv = [], [], []
        self.wres = []

    def build(self, input_shape):
        m = int(input_shape[1])  # slot num
        k = int(input_shape[2])  # feature embedding size
        t = self._att_t
        h = self._att_head
        self.k = k
        self.m = m
        for j in range(self._att_stack):
            self.wq.append([])
            self.wk.append([])
            self.wv.append([])
            for i in range(self._att_head):
                self.wq[j].append(self.add_weight(name='att_layer_' + str(j) + '_wq_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001)))
                self.wk[j].append(self.add_weight(name='att_layer_' + str(j) + '_wk_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001)))
                self.wv[j].append(self.add_weight(name='att_layer_' + str(j) + '_wv_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001)))
            self.wres.append(self.add_weight(name='att_layer_' + str(j) + '_wres_',
                                             shape=[m * k, m * t * h],
                                             dtype=tf.float32,
                                             initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001)))

    def call(self, inputs, training):
        outputs = []
        x = inputs
        for j in range(self._att_stack):
            concat_input = []
            for i in range(self._att_head):
                # x (batch, m, k); wq (k, t)
                q = tf.tensordot(x, self.wq[j][i], axes=[2, 0], name="q")  # (batch, m, t)
                _k = tf.tensordot(x, self.wk[j][i], axes=[2, 0], name="k")  # (batch, m, t)
                v = tf.tensordot(x, self.wv[j][i], axes=[2, 0], name="v")  # (batch, m, t)
                z1 = tf.matmul(q, _k, transpose_b=True)  # (batch, m, m)
                z2 = z1 * (1 / tf.math.sqrt(tf.cast(self.k, tf.float32)))  # (batch, m, m)
                z3 = tf.nn.softmax(z2, axis=2, name="attention_softmax")  # (batch, m, m)
                # print sofmax
                # z3 = tf.Print(z3, [z3], summarize=1000, message="softmax ouput: ")
                z4 = tf.matmul(z3, v)  # (batch, m, t)
                z5 = tf.reshape(z4, shape=[-1, self.m * self._att_t])  # (batch, m * t)
                concat_input.append(z5)
            mhsa = tf.concat(concat_input, axis=1)  # (batch, m * t * h)

            # self.add_param(wres)
            x_r2 = tf.reshape(x, shape=[-1, self.m * self.k])  # (batch, m * k)
            z1 = tf.matmul(x_r2, self.wres[j]) + mhsa  # (batch, m * t * h)
            z_r2 = tf.nn.relu(z1)  # (batch, m * t * h)
            z_r3 = tf.reshape(z_r2, shape=[-1, self.m, self._att_t * self._att_head])  # (batch, m, t * h)
            outputs.append(z_r2)
            x = z_r3
        attention_output = tf.concat(outputs, axis=1, name="attention_embedding")
        return attention_output


class FM(K.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions \
        without linear term and bias.

    References:
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def call(self, inputs, training):
        """
        Invoke this FM layer to output second-order interacted features.

        Args:
            inputs (tf.Tensor): 3-D tensor, often with shape of (batch_size, feature_num, feature_dim)

        Raises:
            ValueError: Raised if `inputs` is invalid.

        Returns:
            tf.Tensor: 2-D tensor with shape of (batch_size, 1).
        """
        sum_of_slots = tf.reduce_sum(inputs, axis=0)
        square_of_sum = tf.square(sum_of_slots)
        square_of_slots = tf.square(inputs)
        sum_of_square = tf.reduce_sum(square_of_slots, axis=0)
        output = 0.5 * tf.subtract(square_of_sum, sum_of_square)
        return output


class MLP(BaseLayer):
    """The Multi Layer Percetron.
    """

    def __init__(self,
                 hidden_units: List[int],
                 activation: str = 'relu',
                 dropout_rate: float = 0.,
                 use_bn: bool = False,
                 use_ln: bool = False,
                 output_activation: str = None,
                 **kwargs):
        """Inits the mlp layer.

        Args:
            hidden_units (List[int]): Hidden units of each layer.
            activation (str): Activation function name of the inner layers, except the output layer.
            dropout_rate (float): Droput rate in inner layers. Default to 0.
            use_bn (bool): Whether use batch normalization in inner layers.
            output_activation (str): Activation function name of the output layer. Default to None.
        """

        super(MLP, self).__init__(**kwargs)

        self._hidden_units = hidden_units
        self._activations = [activation] * (len(hidden_units) - 1) + [output_activation]
        self._dropout_rate = dropout_rate
        self._use_bn = use_bn
        self._use_ln = use_ln
        self._output_activation = output_activation

        # Build dense, batchnormalization and droput layers.
        self._dense_layers = [K.layers.Dense(units=unit, activation=act)
                              for unit, act in zip(hidden_units, self._activations)]
        if use_bn:
            self._bn_layers = [K.layers.BatchNormalization() for _ in range(len(hidden_units) - 1)]
        if use_ln:
            self._ln_layers = [K.layers.LayerNormalization(axis=-1) for _ in range(len(hidden_units) - 1)]
        if dropout_rate > 0.:
            self._dropout_layers = [K.layers.Dropout(dropout_rate) for _ in range(len(hidden_units) - 1)]

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        outputs = inputs
        for i in range(len(self._hidden_units) - 1):
            outputs = self._dense_layers[i](outputs)
            if self._use_bn:
                outputs = self._bn_layers[i](outputs, training)
            if self._use_ln:
                outputs = self._ln_layers[i](outputs)
            if self._dropout_rate > 0.:
                outputs = self._dropout_layers[i](outputs, training)
        outputs = self._dense_layers[-1](outputs)
        return outputs

    def get_config(self):
        config = {
            'activation': self._activations,
            'hidden_units': self._hidden_units,
            'use_bn': self._use_bn,
            'use_ln': self._use_ln,
            'dropout_rate': self._dropout_rate,
            'output_activation': self._output_activation
        }
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
