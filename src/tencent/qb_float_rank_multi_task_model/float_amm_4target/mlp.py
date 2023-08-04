"""A common MLP layer"""

from typing import List

import tensorflow as tf
import tensorflow.keras as K

from model_zoo.layers.base_layer import BaseLayer


class MLP(BaseLayer):
    """The Multi Layer Percetron.
    """

    def __init__(self,
                 hidden_units: List[int],
                 activation: str = 'relu',
                 dropout_rate: float = 0.,
                 use_bn: bool = False,
                 output_activation: str = None,
                 **kwargs):
        """Inits the mlp layer.

        Args:
            hidden_units (List[int]): Hidden units of each layer.
            activation (str): Activation function name of the inner layers, except the output layer.
            dropout_rate (float): Droput rate in inner layers. Default to 0.
            use_bn (bool): Whether use batchnormalization in inner layers.
            output_activation (str): Activation function name of the output layer. Default to None.
        """

        super(MLP, self).__init__(**kwargs)

        self._hidden_units = hidden_units
        self._activations = [activation] * (len(hidden_units) - 1) + [output_activation]
        self._dropout_rate = dropout_rate
        self._use_bn = use_bn
        self._output_activation = output_activation

        # Build dense, batchnormalization and droput layers.
        self._dense_layers = []
        self._dense_layers.append(K.layers.Dense(units=hidden_units[0]))
        for unit, act in zip(hidden_units[1:], self._activations[1:]):
            self._dense_layers.append(K.layers.Dense(units=unit, activation=act))
        self.amm_dense_layers = [K.layers.Dense(hidden_units[0], name="{}_amm_dense_layer".format(self._name)) for i in range(5)]
        if use_bn:
            self._bn_layers = [K.layers.BatchNormalization() for _ in range(len(hidden_units) - 1)]
        if dropout_rate > 0.:
            self._dropout_layers = [K.layers.Dropout(dropout_rate) for _ in range(len(hidden_units) - 1)]

    def call(self, inputs: tf.Tensor, amm_feat: tf.Tensor, training: bool) -> tf.Tensor:
        outputs = inputs
        for i in range(len(self._hidden_units) - 1):
            if i == 0:
                if isinstance(amm_feat, list):
                    amm_feat_output = tf.add_n([self.amm_dense_layers[i](amm_feat[i]) for amm_input in range(len(amm_feat))])
                else:
                    amm_feat_output = self.amm_dense_layers[0](amm_feat)
                outputs = self._dense_layers[i](outputs)
                outputs = tf.nn.relu(amm_feat_output + outputs)
            else:
                outputs = self._dense_layers[i](outputs)
            if self._use_bn:
                outputs = self._bn_layers[i](outputs, training)
            if self._dropout_rate > 0.:
                outputs = self._dropout_layers[i](outputs, training)
        outputs = self._dense_layers[-1](outputs)
        return outputs

    def get_config(self):
        config = {
            'activation': self._activations,
            'hidden_units': self._hidden_units,
            'use_bn': self._use_bn,
            'dropout_rate': self._dropout_rate,
            'output_activation': self._output_activation
        }
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
