# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List

import tensorflow as tf
import tensorflow.keras as K

from model_zoo.layers.base_layer import BaseLayer
# from model_zoo.layers import MLP
from .mlp import MLP


class Gate(K.layers.Layer):
    def __init__(self, slot_num, hidden_unit, decay_weight=1e-4, **kwargs):
        super(Gate, self).__init__(**kwargs)
        self.decay_weight = decay_weight
        self.slot_weights = self.add_weight(
            name="gate_slot_weights",
            shape=(slot_num, hidden_unit),
            initializer=K.initializers.Ones(),
            trainable=True
        )

    def call(self, inputs, **kwargs):
        output = tf.sigmoid(tf.multiply(inputs * self.decay_weight, self.slot_weights)) * 2 - 1.0
        # output = tf.clip_by_value(tf.multiply(inputs/self.threshold, self.slot_weights), 0.0, 1.0)
        return output


class ShowClickLayer(BaseLayer):
    """
    ShowClick Layer module.

    Args:

    """

    def __init__(self,
                 slots: List[str],
                 filter_slots: List[str],
                 replace_slot: int,
                 labels: List[str],
                 dim: int,
                 task_names: List[str],
                 norm_type: str = None,
                 dropout_rate: float = 0.,
                 use_share: bool = False,
                 debug_mode: bool = False,
                 decay_weight: float = 1e-4,
                 **kwargs
                 ):

        super(ShowClickLayer, self).__init__(**kwargs)

        self.slots = slots
        self.filter_slots = filter_slots
        self.replace_slot = replace_slot
        self.labels = labels
        self.dim = dim
        self.task_names = task_names
        self.norm_type = norm_type
        self.dropout_rate = dropout_rate
        self.use_share = use_share
        self.debug_mode = debug_mode
        self.decay_weight = decay_weight

        # gate
        self.gate = Gate(len(slots), dim - 2, self.decay_weight)
        # wide layer
        self.dnn_layers = {}
        for name in task_names:
            # self.dense_layers[name] = K.layers.Dense(units=1, activation=None)
            self.dnn_layers[name] = MLP(
                name=name,
                hidden_units=[256, 128, 1],
                norm_type=self.norm_type,
                dropout_rate=self.dropout_rate,
                activation='relu',
                output_activation=None)

    def call(self, inputs, training) -> (List[tf.Tensor], List[tf.Tensor]):
        """
        Args:
            inputs: A 3-D ``tf.Tensor`` with shape of ``(batch_size, field_size, emb_size)``.
            training: A ``bool`` indicating whether the call is meant for training or inference.

        Returns:
            deep_output: Tensor, Tensor shape: [batch_size, (dim - 2)* len(slots) * 2]
            wide_outputs: List[Tensor], Tensor shape: [batch_size, 1], length: len(task_names)
        """
        # get show click embedding
        if self.use_share:
            show_click_features = inputs['show_click_handcraft_w']
        else:
            feature_names = [f'show_click_handcraft_{slot}' if slot not in set(self.filter_slots)
                             else f'show_click_handcraft_{self.replace_slot}' for slot in self.slots]
            show_click_features = tf.stack([inputs[name] for name in feature_names], axis=1)
        # self.perturb_on.assign(tf.math.logical_not(self.perturb_on))
        print("show_click_features shape:", show_click_features.shape)  # [bs, 185, 12]
        if self.debug_mode:
            tf.print("slots:", self.slots, "show_click_features:", show_click_features, summarize=-1)
        show_click_statistic_arr = tf.split(show_click_features, num_or_size_splits=self.dim, axis=-1)
        show_statistic = show_click_statistic_arr[self.labels.index('show')]
        time_statistic = show_click_statistic_arr[self.labels.index('stime')]
        # show_click_emb_outputs = [tf.math.log(show_statistic + 1.0)]
        show_click_emb_list = []
        for index, label in enumerate(self.labels):
            if label in {'show', 'stime'}:
                continue
            click_statistic = show_click_statistic_arr[index]
            # operations among statistic variables, e.g., sum them
            if label == 'splaytime':
                statistic_new = click_statistic / (time_statistic + 1.0)
            else:
                statistic_new = click_statistic / (show_statistic + 1.0)
            # weight = tf.clip_by_value(show_statistic * 1.0/self.show_threshold, 0.0, 1.0)
            # statistic_new = tf.clip_by_value(weight * statistic_new, 0.0, 3.0)
            statistic_new = tf.clip_by_value(statistic_new, 0.0, 3.0)
            show_click_emb_list.append(statistic_new)
            # show_click_emb_list.append(tf.math.log(statistic_new + 1.0))
        # show_click_emb = tf.squeeze(tf.concat(show_click_emb_list, axis=1), axis=-1)
        # deep_output = tf.stop_gradient(show_click_emb, name="show_click_emb")
        show_click_emb = tf.stop_gradient(tf.concat(show_click_emb_list, axis=-1), name='show_click_emb')
        print("show_click_emb shape:", show_click_emb.shape)    # [bs, 185, 10]
        ctr_static_decayed = self.gate(show_statistic) * show_click_emb
        shape = show_click_emb.get_shape().as_list()
        print('ctr_static_decayed shape:', ctr_static_decayed.shape)
        deep_output = tf.reshape(ctr_static_decayed, [-1, shape[1]*shape[2]])
        print("deep_output shape:", deep_output.shape)    # [None, 1850]

        wide_outputs = []
        for name in self.task_names:
            output = self.dnn_layers[name](inputs=deep_output, training=training)
            wide_outputs.append(output)

        return deep_output, wide_outputs

    # show click
    # 验证show click功能的正确性，单独筛选出前2列和无量的show click比较。用户实际使用时，不需要加这部分代码。
    def check(self, inputs):
        for slot in self.slots:
            embedding = inputs['show_click_api_{}'.format(slot)]
            show_click_numerous_1 = tf.slice(embedding, [0, 0], [-1, 2])
            show_click_handcraft = inputs["show_click_handcraft_{}".format(slot)]
            show_click_numerous_2 = tf.slice(show_click_handcraft, [0, 0], [-1, 2])
            tf.debugging.assert_equal(show_click_numerous_1, show_click_numerous_2,
                                      message=f"show_click not equal, slot:{slot}", summarize=-1)
        assert self.labels[0] == 'show' and self.labels[1] == 'click', 'index 0 and 1 must be show and click!'

    def get_config(self):
        config = {
            'slots': self.slots,
            'filter_slots': self.filter_slots,
            'replace_slot': self.replace_slot,
            'labels': self.labels,
            'show_click_emb_dim': self.dim,
            'task_names': self.task_names,
            'norm_type': self.norm_type,
            'dropout_rate': self.dropout_rate,
            'use_share': self.use_share,
            'decay_weight': self.decay_weight,
        }
        base_config = super(ShowClickLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
