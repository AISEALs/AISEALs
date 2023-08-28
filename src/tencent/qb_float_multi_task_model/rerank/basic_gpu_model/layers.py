# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as K

from model_zoo.layers import MMoE
from model_zoo.layers import MLP

from typing import List


## same with precise rank
def get_label(pt, vt):
    pr = pt / (vt + 1e-8)
    pr = tf.clip_by_value(pr, 0, 1)
    return pr


class MultiHeadSelfAttention(K.layers.Layer):
    """A implementation of the Multi-head self-attention module.

    """

    def __init__(self, attention_embedding_size, num_heads, num_blocks, prefix, **kwargs):
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
        self.prefix = prefix
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
                self.wq[j].append(self.add_weight(name=self.prefix + '_mhsar_' + str(j) + '_wq_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001)))
                self.wk[j].append(self.add_weight(name=self.prefix + '_mhsar_' + str(j) + '_wk_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001)))
                self.wv[j].append(self.add_weight(name=self.prefix + '_mhsar_' + str(j) + '_wv_' + str(i),
                                                  shape=[k, t],
                                                  dtype=tf.float32,
                                                  initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001)))
            self.wres.append(self.add_weight(name=self.prefix + '_mhsar_' + str(j) + '_wres_',
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
        attention_output = tf.concat(outputs, axis=1, name=self.prefix + "_attention_embedding")
        return attention_output


class SENet(K.layers.Layer):
    def __init__(self, filed_num, embedding_dim, squeeze_dim, **kwargs):
        super(SENet, self).__init__(**kwargs)
        self.filed_num = filed_num
        self.embedding_dim = embedding_dim
        self.squeeze_dim = squeeze_dim
        self.A1 = K.layers.Dense(units=squeeze_dim, activation='relu', kernel_initializer='glorot_uniform', name='A1')
        self.A2 = K.layers.Dense(units=filed_num, activation='sigmoid', kernel_initializer='glorot_uniform', name='A2')
        pass

    def call(self, inputs, training):
        input_layer = tf.reshape(inputs, shape=[-1, self.filed_num,
                                                self.embedding_dim])  # shape:[batchsize, slot_num, emb_size]
        Z = tf.reduce_mean(input_layer, axis=2)
        tmp = self.A1(Z)
        tmp2 = self.A2(tmp)
        output = tf.multiply(input_layer, tf.expand_dims(tmp2, axis=2))
        output = tf.reshape(output, shape=[-1, self.filed_num * self.embedding_dim])
        return output


class DeepLayers(K.layers.Layer):
    def __init__(self, layersizes, model_name, **kwargs):
        super(DeepLayers, self).__init__(**kwargs)
        self.layersizes = layersizes
        self.model_name = model_name
        self.nets = []
        for num in range(0, len(self.layersizes)):
            net = K.layers.Dense(units=self.layersizes[num], activation='selu',
                                      kernel_initializer='glorot_uniform', name="%s_fc_%d" % (self.model_name, num))
            self.nets.append(net)

    def call(self, inputs, training):
        for net in self.nets:
            inputs = net(inputs)
        return inputs


class MMOE(K.layers.Layer):
    def __init__(self, model_name, debug, **kwargs):
        super(MMOE, self).__init__(**kwargs)
        self.layersizes = [1024, 512, 256, 128]
        self.num_experts = 4
        self.num_tasks = 4
        self.model_name = model_name
        self.debug = debug

        self.experts = []
        for num in range(self.num_experts):
            expert = DeepLayers(layersizes=self.layersizes, model_name="%s_expert_%d" % (self.model_name, num))
            self.experts.append(expert)

        self.tower_layers = []
        for num in range(0, self.num_tasks):
            tower_layer1 = K.layers.Dense(units=self.num_experts, activation='relu',
                                          kernel_initializer='glorot_uniform',
                                          name="%s_weight%d" % (self.model_name, num))
            tower_layer2 = K.layers.Activation('softmax', name="%s_gate%d" % (self.model_name, num))
            self.tower_layers.append((tower_layer1, tower_layer2))

    def call(self, inputs, training):
        expert_layers = []
        for expert in self.experts:
            expert_layer = expert(inputs, training)
            expert_layers.append(expert_layer)
            if self.debug:
                print(tf.shape(expert_layer))
                print("-----")
        ## experts outputs
        expert_concat_layer = tf.stack(expert_layers, axis=2, name="%s_expert_concat_layer" % self.model_name)

        ## tower output
        towers = []
        for tower_layer1, tower_layer2 in self.tower_layers:
            weight = tower_layer1(inputs)
            gate = tower_layer2(weight)
            if self.debug:
                print("gate shape:" % gate.shape)
            tower = tf.multiply(expert_concat_layer, tf.expand_dims(gate, axis=1))
            tower = tf.reduce_sum(tower, axis=2)
            if self.debug:
                print("tower shape:" % tower.shape)
            ## reshape to output size
            tower = tf.reshape(tower, [-1, self.layersizes[-1]])
            if self.debug:
                print("tower reshape:" % tower.shape)
            towers.append(tower)
        return towers


class MMoELayer(MMoE):
    def __init__(self,
                 task_names: List[str],
                 expert_num: int,
                 expert_hidden_sizes: List[int],
                 tower_hidden_sizes: List[int],
                 **kwargs):
        super(MMoELayer, self).__init__(task_names, expert_num, expert_hidden_sizes, tower_hidden_sizes, **kwargs)

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
            output_activation=None)
        return mlp
