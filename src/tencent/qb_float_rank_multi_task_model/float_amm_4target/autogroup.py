# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict

import tensorflow as tf
import tensorflow.keras as K

from model_zoo.layers.base_layer import BaseLayer


class AutoGroupLayer(BaseLayer):
    """
    AutoGroup Layer module.

    Use differentiable neural architecture search techniques to automatically select features and make interactions.

    References:
        `AutoGroup: Automatic Feature Grouping for Modelling Explicit High-Order Feature Interactions in CTR Prediction.
        <https://doi.org/10.1145/3397271.3401082>`_

    Args:
        slot_num: An ``int``, the number of slots.
        group_num: An ``int``, the number of cross features (groups) to be selected.
        order: An ``int``, the order of interaction.
        temperature: A ``float``, the temperature parameter in Gumbel-softmax.
    """

    def __init__(self,
                 slot_num: int,
                 group_num: int,
                 order: int,
                 temperature: float = 1.0,
                 **kwargs
                 ):

        if order < 1:
            raise ValueError("`order` should be a positive integer.")
        super(AutoGroupLayer, self).__init__(**kwargs)
        self.slot_num = slot_num
        self.group_num = group_num
        self.order_raw = order
        self.order = tf.constant(order, dtype=tf.float32)
        self.temperature = temperature
        self.epsilon = 1e-10
        # parameters

        self.structure_logits = self.add_weight(
            name=self.name + "_structure_logits",
            shape=(self.slot_num, self.group_num, 2),
            initializer=K.initializers.RandomUniform(minval=-0.001, maxval=0.001),
            trainable=True
        )
        self.slot_weights = self.add_weight(
            name=self.name + "_slot_weights",
            shape=(self.slot_num, self.group_num),
            initializer=K.initializers.GlorotUniform(),
            trainable=True
        )
        self.perturb_on = self.add_weight(name=self.name + "_perturb_on",
                                          dtype=tf.bool,
                                          shape=[1],
                                          initializer=tf.keras.initializers.Ones(),
                                          trainable=False)

        # other properties
        self.layer_normalization = tf.keras.layers.LayerNormalization(axis=[1, 2])
        self._choice_matrix = tf.constant([[1.0], [0.0]])
        self.selections = None

    def call(self, inputs, training) -> tf.Tensor:
        """Invoke AutoGroup layer to calculate features in each group and make interactions.

        Args:
            inputs: A 3-D ``tf.Tensor`` with shape of ``(batch_size, field_size, emb_size)``.
            training: A ``bool`` indicating whether the call is meant for training or inference.

        Returns:
            A ``tf.Tensor``. interacted features in each group.
            The shape is ``(batch, group_num, embed_dim)`` if order > 1, and
             ``(batch, slot_num, embed_dim)`` if order = 1.
        """
        embed_matrix = self.layer_normalization(inputs)
        if self.order_raw == 1:
            return embed_matrix

        self.selections = self._differentiable_sampling(training)
        interacted_features = self._make_interactions(embed_matrix)
        return interacted_features

    def _differentiable_sampling(self, training) -> tf.Tensor:
        """Use Gumbel-Softmax trick to take argmax, while keeping differentiate w.r.t soft sample y.
        Note that this function is different from the original implementation in Huawei's vega project.
        This implementation further restricts the number in each group/group being equal to the order.

        Returns:  A 2-D ``tf.Tensor``, the selection masks of groups.
         The value of the selected feature will be 1, and will be 0 otherwise.
        """
        self.perturb_on.assign(tf.math.logical_not(self.perturb_on))
        perturbed_flag = tf.math.logical_and(training, self.perturb_on)
        gumbel_logits = self._make_gumbel_logits()
        logits = tf.where(perturbed_flag, gumbel_logits, self.structure_logits)
        probs = tf.nn.softmax(logits / self.temperature, axis=-1)

        # Select slots with top-k (k=order) probability in each group as the interacted slots.
        selected_probs = tf.squeeze(
            tf.linalg.matmul(probs, self._choice_matrix)
        )
        selected_probs_per_group = tf.transpose(selected_probs, perm=[1, 0])  # [group_num, slot_num]
        _, top_k_indices = tf.math.top_k(selected_probs_per_group, k=self.order_raw)
        one_hots = tf.one_hot(top_k_indices, self.slot_num)
        hard_top_k_selections = tf.reduce_max(one_hots, axis=1)
        hard_top_k_selections = tf.transpose(hard_top_k_selections, perm=[1, 0])  # [slot_num, group_num]
        soft_top_k_selections = selected_probs
        trainable_selections = tf.stop_gradient(hard_top_k_selections - soft_top_k_selections) + soft_top_k_selections
        return trainable_selections

    def _make_interactions(self, embed_matrix) -> tf.Tensor:
        """Make interaction with high order FM formula.
        See paper https://doi.org/10.1145/3397271.3401082 for more detail.
        """
        embed_sum = tf.linalg.matmul(
            tf.transpose(embed_matrix, perm=[0, 2, 1]),
            tf.math.multiply(self.selections, self.slot_weights)
        )
        embed_pow_of_sum = tf.math.pow(embed_sum, self.order)

        embed_sum_of_pow = tf.linalg.matmul(
            tf.math.pow(
                tf.transpose(embed_matrix, perm=[0, 2, 1]), self.order
            ),
            tf.math.pow(
                tf.math.multiply(self.selections, self.slot_weights), self.order
            ),
        )

        interacted_features = tf.transpose(
            embed_pow_of_sum - embed_sum_of_pow, perm=[0, 2, 1]
        )
        # [batch, group_num, embed_dim]
        return interacted_features

    def _make_gumbel_logits(self) -> tf.Tensor:
        noise = tf.random.uniform(shape=(self.slot_num, self.group_num, 2))
        gumbel_logits = self.structure_logits - tf.math.log(
            -tf.math.log(noise + self.epsilon) + self.epsilon
        )
        return gumbel_logits

    def get_features_in_groups(self) -> Dict[str, tf.Tensor]:
        features_in_groups = {}
        if self.order_raw == 1:
            for i in range(self.slot_num):
                group_name = "order_{}_group_{}".format(self.order_raw, i)
                features_in_groups[group_name] = tf.constant([i])
        else:
            selections = tf.split(self.selections, num_or_size_splits=self.group_num, axis=1)
            for i, selection in enumerate(selections):
                group_name = "order_{}_group_{}".format(self.order_raw, i)
                features_in_groups[group_name] = tf.reshape(tf.where(tf.squeeze(selection) > 0.5), [-1])
        return features_in_groups

    def get_config(self):
        config = {
            'slot_num': self.slot_num,
            'group_num': self.group_num,
            'order': self.order_raw,
            'temperature': self.temperature
        }
        base_config = super(AutoGroupLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))