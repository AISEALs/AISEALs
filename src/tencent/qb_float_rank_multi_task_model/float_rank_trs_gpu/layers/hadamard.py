# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as K

from model_zoo.layers.base_layer import BaseLayer


class HadamardLayer(BaseLayer):
    """
    Hadamard Layer module.

    Args:
        hadamard_slots: An Dict with (k, v) format: (hadamard_slot: [first_slot, second_slot])
        task_names: An List with task. eg: ['ratio', 'skip', 'is_finish']
    """

    def __init__(self,
                 hadamard_slots: Dict[int, tuple],
                 task_names: List[str],
                 **kwargs
                 ):

        super(HadamardLayer, self).__init__(**kwargs)
        self.hadamard_slots = hadamard_slots
        self.task_names = task_names

        # wide layer
        self.dense_layers = {}
        for name in task_names:
            self.dense_layers[name] = K.layers.Dense(units=1, activation=None)

    def call(self, inputs, training) -> (List[tf.Tensor], List[tf.Tensor]):
        """Invoke Hadamard layer to optimize features storage in each group.

        Args:
            inputs: A dict of tensors, tensor shape: [batch_size, emb_dim]
            training: A ``bool`` indicating whether the call is meant for training or inference.

        Returns:
            deep_outputs: List[Tensor], Tensor shape: [batch_size, emb_dim], length: len(hadamard_slots)
            wide_outputs: List[Tensor], Tensor shape: [batch_size, 1], length: len(task_names)
        """
        deep_outputs = []
        for slot_id, split_slots in self.hadamard_slots.items():
            hadamard_ouput = inputs[f'sparse_w_{split_slots[0]}'] * inputs[f'sparse_w_{split_slots[1]}']
            deep_outputs.append(hadamard_ouput)

        wide_outputs = []
        for name in self.task_names:
            # output = tf.stack(deep_outputs, axis=1)
            # output = self.dense_layers[name](tf.reduce_sum(output, axis=-1))
            output = self.dense_layers[name](tf.concat(deep_outputs, axis=1))
            wide_outputs.append(output)

        return deep_outputs, wide_outputs

    def get_config(self):
        config = {
            'task_names': self.task_names,
            'hadamard_slots': self.hadamard_slots,
        }
        base_config = super(HadamardLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
