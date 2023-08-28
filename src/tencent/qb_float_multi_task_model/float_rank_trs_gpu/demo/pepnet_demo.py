#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numerous
import tensorflow as tf
import tensorflow.keras as K

from model_zoo.inputs import Inputs
from model_zoo.models import NumerousModel
from model_zoo.layers.base_layer import BaseLayer

FEATURE_NAMES = ['gender', 'age', 'occupation', 'zipcode', 'title', 'genres', 'timestamp', 'uid', 'itemid',
                 'uid_ppnet', 'itemid_ppnet', 'domain_id']
SLOT_IDS = [100, 101, 102, 103, 104, 105, 106, 110, 120, 110, 120, 101]
TASK_NAMES = ["task1", "task2"]


class GateNN(K.layers.Layer):
    def __init__(self, hidden_unit, output_unit, name, idx, activation='relu', **kwargs):
        super(GateNN, self).__init__(**kwargs)
        self.hidden_layer = tf.keras.layers.Dense(hidden_unit,
                                                  activation=activation,
                                                  name=self.name + str(idx) + "_0")
        self.output_layer = tf.keras.layers.Dense(output_unit,
                                                  activation='sigmoid',
                                                  name=self.name + str(idx) + "_1")

    def call(self, inputs, **kwargs):
        hidden = self.hidden_layer(inputs)
        output = 2 * self.output_layer(hidden)
        return output


class PEPNet(BaseLayer):
    """
    Parameter and Embedding Personalized Network: Adjust the output of each layer \
    of the deep network based on user personalization.
      Input shape
        - List of 3 tensors [domain_embedding, tower_inputs, auxiliary_embedding], \
        each tensor is a nD tensor with shape: ``(batch_size, ..., input_dim)``.\
            The most common situation would be a 2D tensor with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_units[-1])``.\
            For instance, for a 2D input with shape ``(batch_size, input_dim)``,\
                the output would have shape ``(batch_size, hidden_units[-1])``.

      Args:
            tower_num: A ``int``, the tower num 
            hidden_units: A ``list`` of ``int``, the layer number and units in each layer.

            gate_units: A ``int``, the units in GateNN layer
            activation: A ``str``, Activation function used for DNN.
            gate_activation: A ``str``, Activation function used for GateNN.
            output_activation: A ``str``, Activation function to used for output layer.
            use_bn: A ``bool``, Whether use BatchNormalization on input embedding for DNN.
            use_gate_bn: A ``bool``, Whether use BatchNormalization on input embedding for GateNN.
            weight_last_layer: A ``bool``, Whether apply ppnet on last layer of MLP.
            **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, tower_num, hidden_units, domian_gate_units, gate_units, name, activation='relu',
                 gate_activation='relu', output_activation=None, use_bn=True, use_gate_bn=True, weight_last_layer=False,
                 **kwargs):
        super(PEPNet, self).__init__(name=name, **kwargs)
        self.tower_num = tower_num
        self.hidden_units = hidden_units
        self.domian_gate_units = domian_gate_units
        self.gate_units = gate_units
        self.activation = activation
        self.gate_activation = gate_activation
        self.output_activation = output_activation
        self.use_gate_bn = use_gate_bn
        self.use_bn = use_bn
        self.weight_last_layer = weight_last_layer
        self.dense_layers = []
        if self.use_bn:
            self.bn_layer = [K.layers.BatchNormalization() for _ in range(tower_num)]
        if self.use_gate_bn:
            self.gate_bn_layer = [K.layers.BatchNormalization() for _ in range(tower_num)]
        for t in range(self.tower_num):
            self.dense_layers.append([])
            for i in range(len(self.hidden_units) - 1):
                layer = tf.keras.layers.Dense(units=self.hidden_units[i],
                                              activation=self.activation,
                                              name=self.name + "_dnn_" + str(t) + "_" + str(i))
                self.dense_layers[t].append(layer)
            layer = tf.keras.layers.Dense(units=self.hidden_units[len(self.hidden_units) - 1],
                                          activation=self.output_activation,
                                          name=self.name + "_dnn_" + str(t) + "_" + str(len(self.hidden_units) - 1))
            self.dense_layers[t].append(layer)
        self.gate_nn = []

    def build(self, input_shape):
        nn_embedding_size = input_shape[1][-1]
        self.domain_gate = GateNN(self.domian_gate_units,
                                  nn_embedding_size,
                                  self.name + "_domain_gate",
                                  0,
                                  self.gate_activation
                                  )
        self.gate_nn.append(GateNN(self.gate_units,
                                   nn_embedding_size,
                                   self.name + "_embedding_gate",
                                   0,
                                   self.gate_activation))
        for i, unit_num in enumerate(self.hidden_units[:-1]):
            self.gate_nn.append(GateNN(self.gate_units,
                                       unit_num,
                                       self.name + "_nn_gate",
                                       i + 1,
                                       self.gate_activation))
        if self.weight_last_layer:
            self.gate_nn.append(GateNN(self.gate_units, self.hidden_units[-1], self.gate_activation))
        super(PEPNet, self).build(input_shape)

    def call(self, inputs, training=True, **kwargs):  # pylint: disable=arguments-differ
        ret_out = []
        domain_embedding, tower_inputs, auxiliary_embedding = inputs
        tower_gate = self.domain_gate(domain_embedding)
        tower_inputs = tower_inputs * tower_gate

        for t in range(self.tower_num):
            main_embedding = tf.stop_gradient(tower_inputs)
            gate_nn_input = tf.concat((auxiliary_embedding, main_embedding), axis=1)
            if self.use_bn:
                tower_inputs = self.bn_layer[t](tower_inputs)
            if self.use_gate_bn:
                gate_nn_input = self.gate_bn_layer[t](gate_nn_input)
            hidden = tower_inputs
            for i in range(len(self.hidden_units)):
                gate = self.gate_nn[i](gate_nn_input)
                hidden = hidden * gate
                hidden = self.dense_layers[t][i](hidden)
            if self.weight_last_layer:
                hidden = self.gate_nn[-1](gate_nn_input) * hidden
            output = hidden
            ret_out.append(output)
        return ret_out

    def get_config(self):
        config = {'activation': self.activation, 'gate_activation': self.gate_activation,
                  'output_activation': self.output_activation, 'hidden_units': self.hidden_units,
                  'gate_units': self.gate_units, 'name': self.name, 'use_bn': self.use_bn,
                  'use_gate_bn': self.use_gate_bn, 'weight_last_layer': self.weight_last_layer}
        base_config = super(PEPNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PEPNetModel(NumerousModel):
    def __init__(self, name):
        super().__init__(name)
        self.losser = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        for task_name in TASK_NAMES:
            self.add_metric(name=f"{task_name}_auc", metric=numerous.metrics.AUC(name=f"{task_name}_auc"))
        self.set_optimizer(numerous.optimizers.Adam(learning_rate=0.0001))

        self.set_saver(
            numerous.train.Saver(
                dumper_config=numerous.train.DumperConfig(
                    dump_policy=numerous.train.DumpPolicy.DIR,
                    dump_interval=1,
                    dump_type=numerous.train.DumpType.FULL,
                ),
                expire_config=numerous.train.FeatureExpireConfig(
                    expire_time=3600 * 24 * 14,
                )
            )
        )

        self.pepnet = PEPNet(
            tower_num=2,
            hidden_units=[100, 100, 1],
            domian_gate_units=50,
            gate_units=100,
            name='pepnet'
        )

    def build_inputs(self):
        emb_initializer = numerous.initializers.Uniform(minval=-0.001, maxval=0.001)
        emb_optimizer = numerous.optimizers.Adam(learning_rate=0.0001)
        inputs = Inputs(optimizer=emb_optimizer, initializer=emb_initializer)

        for feature_name, slot_id in zip(FEATURE_NAMES, SLOT_IDS):
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=8,
                optimizer=emb_optimizer,
                initializer=emb_initializer,
                combiner=numerous.layers.Sum()
            )
        return inputs

    def call(self, inputs, training):
        inputs = self.preprocess_inputs(inputs)  # process the features into a dictionary, optional
        # Get feature by feature_name in `inputs`
        sparse_feature_names = ['gender', 'age', 'occupation', 'zipcode', 'title', 'genres', 'timestamp',
                                'uid', 'itemid']
        id_feature_names = ['uid_ppnet', 'itemid_ppnet']
        id_domain_names = ['domain_id']
        sparse_features = [inputs[feature_name] for feature_name in sparse_feature_names]
        id_features = [inputs[feature_name] for feature_name in id_feature_names]
        domain_features = [inputs[feature_name] for feature_name in id_domain_names]

        nn_embedding = tf.concat(sparse_features, axis=1)
        id_embedding = tf.concat(id_features, axis=1)
        domain_embedding = tf.concat(domain_features, axis=1)
        features = [domain_embedding, nn_embedding, id_embedding]

        predictions = self.pepnet(features, training)
        predictions = [tf.math.sigmoid(pred) for pred in predictions]
        return {name: pred for name, pred in zip(TASK_NAMES, predictions)}

    def compute_loss(self, labels, model_outputs, sample_weights, training):
        label = labels["label"]
        losses = []
        for task_name in TASK_NAMES:
            loss = self.losser(label, model_outputs[task_name])
            losses.append(loss)
            self.get_metric(f"{task_name}_auc").update_state(label, model_outputs[task_name])
        return losses


numerous.cluster.start(numerous.cluster.get_cluster_config(run_mode=numerous.cluster.RunMode.LOCAL))
with numerous.distribute.get_strategy().scope():
    model = PEPNetModel(name="pepnet")
    model.run()
numerous.cluster.stop()
