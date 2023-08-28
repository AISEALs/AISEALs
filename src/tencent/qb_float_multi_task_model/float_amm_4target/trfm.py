"""Defines Transformer model in tf.keras API."""
from typing import List

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dropout
from model_zoo.layers.base_layer import BaseLayer

NEG_INF = -2 ** 32 + 1


class Projection(BaseLayer):
    """Linearly projects a batch of continuously represented sequences of tokens.
    This projection layer operates in either Split mode or Merge mode:
      - Split mode converts the input sequences in the original representation
        into the multi-headed "query", "key" or "value" for the attention
        computation.
        Input: [batch_size(N), seq_len(T), hidden_size(D)]
        Weight: [hidden_size(D), num_heads(H), size_per_head(S)]
        Output: dot([N*T, D], [D, H*S]) reshape ==> [N, T, H, S]
      - Merge mode performs the opposite action of Split, converting the
        multi-headed "value" back to the original representation.
        Input: [batch_size(N), seq_len(T), num_heads(H), size_per_head(S)]
        Weight: [num_heads(H), size_per_head(S), hidden_size(D)]
        Output: dot([N*T, H*S], [H*S, D]) reshape ==> [N, T, D]
    """

    def __init__(self,
                 num_heads,
                 size_per_head,
                 kernel_initializer='glorot_uniform',
                 mode="split", **kwargs):
        """Constructor.
        Args:
          num_heads: int scalar, num of attention heads.
          size_per_head: int scalar, the hidden size of each attention head.
          kernel_initializer: string scalar, the weight initializer.
          mode: string scalar, mode of projection ("split" or "merge").
        """
        super(Projection, self).__init__(**kwargs)
        if mode not in ('split', 'merge'):
            raise ValueError('"mode" must be either "split" or "merge".')
        self._num_heads = num_heads
        self._size_per_head = size_per_head
        self._hidden_size = num_heads * size_per_head
        self._kernel_initializer = kernel_initializer
        self._mode = mode

    def build(self, inputs_shape):
        """Creates weights of this layer.
        Args:
          inputs_shape: tuple of ints or 1-D int tensor, the last element
            corresponds to the depth.
        """
        depth = inputs_shape[-1]
        if depth is None:
            raise ValueError('The depth of inputs must not be None.')

        if self._mode == 'merge':
            kernel_shape = self._num_heads, self._size_per_head, self._hidden_size
        else:
            kernel_shape = self._hidden_size, self._num_heads, self._size_per_head

        self.add_weight(name='{}_kernel'.format(self._name),
                        shape=kernel_shape,
                        initializer=self._kernel_initializer,
                        dtype='float32',
                        trainable=True)
        super(Projection, self).build(inputs_shape)

    def call(self, inputs):
        """Performs the projection.
        Args:
          inputs: float tensor of shape [batch_size, seq_len, num_heads,
            size_per_head] in Merge mode, or float tensor of shape [batch_size,
            seq_len, hidden_size] in Split mode.
        Returns:
          outputs: float tensor of shape [batch_size, seq_len, hidden_size] in
            Merge mode, or float tensor of shape [batch_size, seq_len, num_heads,
            size_per_head] int Split mode.
        """
        kernel = self.trainable_variables[0]
        if self._mode == 'merge':
            outputs = tf.einsum('NTHS,HSD->NTD', inputs, kernel)
        else:
            outputs = tf.einsum('NTD,DHS->NTHS', inputs, kernel)
        return outputs


class Attention(BaseLayer):
    """Multi-headed attention.
    Given a batch of vector-represented query sequences (tensor of shape [
    batch_size, q_seq_len, hidden_size]) and context sequences (tensor of shape
    [batch_size, c_seq_len, hidden_size]), this layer computes a new
    representation of the query sequences by making them discriminatively attend
    to tokens in the context sequences.
    If the query and context happen to be the same, the result ends up being
    "Self Attention" -- the query sequence attends to itself.
    """

    def __init__(self, hidden_size, num_heads, dropout_rate, use_output=True, **kwargs):
        """Constructor.
        Args:
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(Attention, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate
        self._size_per_head = hidden_size // num_heads
        self.use_output = use_output

        self._dense_layer_query = Projection(
            num_heads, self._size_per_head, mode='split', name="{}_dense_layer_query".format(self._name))
        self._dense_layer_key = Projection(
            num_heads, self._size_per_head, mode='split', name="{}_dense_layer_key".format(self._name))
        self._dense_layer_value = Projection(
            num_heads, self._size_per_head, mode='split', name="{}_dense_layer_value".format(self._name))
        if self.use_output:
            self._dense_layer_output = Projection(
                num_heads, self._size_per_head, mode='merge', name="{}_dense_layer_output".format(self._name))
        self._dropout_layer = Dropout(dropout_rate)

    def call(self, query, context, attention_mask, training, causal=False):
        """Computes new representation of query sequences.
        Args:
          query: float tensor of shape [batch_size, q_seq_len, hidden_size],
            query sequences.
          context: float tensor of shape [batch_size, c_seq_len, hidden_size]
            , context sequences.
          attention_mask: float tensor of shape [batch_size, num_heads, q_seq_len,
            c_seq_len], populated with either 0 (for tokens to keep) or 1 (for
            tokens to be masked).
          training: (Optional) bool scalar, True if in training mode.
          cache: (Optional) dict with entries
            'k': tensor of shape [batch_size * beam_width, seq_len, num_heads,
              size_per_head],
            'v': tensor of shape [batch_size * beam_width, seq_len, num_heads,
              size_per_head],
            'tgt_tgt_attention': tensor of shape [batch_size * beam_width,
              num_heads, tgt_seq_len, tgt_seq_len],
            'tgt_src_attention': tensor of shape [batch_size * beam_width,
              num_heads, tgt_seq_len, src_seq_len].
            Must be provided in inference mode when called within decoder layers.
        Returns:
          outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the
            new representation of `query`.
        """

        # [batch_size, q_seq_len, num_heads, size_per_head]
        q = self._dense_layer_query(query)
        q_len = tf.shape(q)[1]

        # [batch_size, c_seq_len, num_heads, size_per_head]
        k = self._dense_layer_key(context)
        v = self._dense_layer_value(context)

        # [batch_size, num_heads, q_seq_len, c_seq_len]
        attention_weights = tf.einsum('NQHS,NCHS->NHQC', q, k)
        attention_weights *= self._size_per_head ** -0.5
        paddings = tf.ones_like(attention_weights) * NEG_INF
        attention_weights = tf.where(attention_mask, attention_weights, paddings)
        if causal:
            length = attention_weights.get_shape().as_list()[-1]
            causal_mask = tf.linalg.band_part(tf.ones([length, length]), -1, 0)
            causal_mask = tf.transpose(causal_mask, perm=[1, 0])  # ATTENTION!!!!!! 逆序顺序
            causal_mask = tf.cast(causal_mask, tf.bool)
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]
            attention_weights = tf.where(causal_mask, attention_weights, paddings)
        attention_weights = tf.nn.softmax(attention_weights, axis=3)
        attention_weights = self._dropout_layer(attention_weights, training)

        # [batch_size, q_seq_len, num_heads, size_per_head]
        outputs = tf.einsum('NHQC,NCHS->NQHS', attention_weights, v)
        if not self.use_output:
            outputs = tf.reshape(outputs, [-1, q_len, self._hidden_size])
        else:
            # [batch_size, q_seq_len, hidden_size]
            outputs = self._dense_layer_output(outputs)
        return outputs


class FeedForwardNetwork(BaseLayer):
    """The Projection layer that consists of a tandem of two dense layers (an
    intermediate layer and an output layer).
    """

    def __init__(self,
                 hidden_size,
                 filter_size,
                 dropout_rate,
                 filter_activation=tf.nn.relu, **kwargs):
        """Constructor.
        Args:
          hidden_size: int scalar, the hidden size of continuous representation,
            which is also the depth of the output dense layer.
          filter_size: int scalar, the depth of the intermediate dense layer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
          filter_activation: callable or string, activation function of the filter
            dense layer. Defaults to ReLU.
        """
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate
        self._filter_activation = filter_activation

        self._dense_layer_filter = tf.keras.layers.Dense(
            filter_size, use_bias=True, activation=filter_activation)
        self._dense_layer_output = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self._dropout_layer = Dropout(dropout_rate)

    def call(self, inputs, training):
        """Performs projection through two dense layers.
        Args:
          inputs: float tensor of shape [batch_size, seq_len, hidden_size], the
            input sequences.
        Return:
          outputs: float tensor of shape [batch_size, seq_len, hidden_size], the
            output sequences.
        """
        outputs = self._dense_layer_filter(inputs)
        outputs = self._dropout_layer(outputs, training)
        outputs = self._dense_layer_output(outputs)
        return outputs


class EncoderLayer(BaseLayer):
    """The building block that makes the encoder stack of layers, consisting of an
    attention sublayer and a feed-forward sublayer.
    """

    def __init__(self, hidden_size, num_heads, filter_size, dropout_rate, **kwargs):
        """Constructor.

        Args:
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(EncoderLayer, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._mha = Attention(hidden_size, num_heads, dropout_rate, name="{}_mha".format(self._name))
        self._layernorm_mha = tf.keras.layers.LayerNormalization(name="{}_mha_ln".format(self._name))
        self._dropout_mha = Dropout(dropout_rate)

        self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self._layernorm_ffn = tf.keras.layers.LayerNormalization(name="{}_ffn_ln".format(self._name))
        self._dropout_ffn = Dropout(dropout_rate)

    def call(self, inputs, padding_mask, training, causal=False):
        """Computes the output of the encoder layer.

        Args:
          inputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            input source sequences.
          padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).

        Returns:
          outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            output source sequences.
        """
        query = reference = self._layernorm_mha(inputs)
        outputs = self._mha(query, reference, padding_mask, training, causal=causal)
        ffn_inputs = self._dropout_mha(outputs, training) + inputs
        outputs = self._layernorm_ffn(ffn_inputs)
        outputs = self._ffn(outputs, training)
        outputs = self._dropout_ffn(outputs, training) + ffn_inputs
        return outputs


class Encoder(BaseLayer):
    """The Encoder that consists of a stack of structurally identical layers."""

    def __init__(
            self, stack_size, hidden_size, num_heads, filter_size, dropout_rate, seq_length, **kwargs):
        """Constructor.

        Args:
          stack_size: int scalar, num of layers in the stack.
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(Encoder, self).__init__(**kwargs)
        self._stack_size = stack_size
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._stack = []
        # self.pos_enc = self.add_weight(
        #     "{}_pos_embeddings".format(self._name),
        #     shape=[seq_length, self._hidden_size],
        #     trainable=True,
        #     initializer=tf.keras.initializers.GlorotUniform())
        self.pos_dropout = Dropout(self._dropout_rate)

        for i in range(self._stack_size):  # 2
            self._stack.append(EncoderLayer(hidden_size,
                                            num_heads,
                                            filter_size,
                                            dropout_rate, name="{}_enc_layer_{}".format(self._name, i)))
        self._layernorm = tf.keras.layers.LayerNormalization(name="{}_enc_output_ln".format(self._name))

    def call(self, inputs, padding_mask, training, causal=False):
        """Computes the output of the encoder stack of layers.

        Args:
          inputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            input source sequences.
          padding_mask: boolean tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 1 (for tokens to keep) or 0 (for tokens to be
            masked).
          training: bool scalar, True if in training mode.

        Returns:
          outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            output source sequences.
        """
        padding_mask = tf.expand_dims(tf.expand_dims(padding_mask, 1), 1)
        # inputs = inputs * (self._hidden_size ** 0.5)
        # inputs += tf.expand_dims(self.pos_enc, 0)  # Len x hidden_size
        # inputs = self.pos_dropout(inputs, training)
        for layer in self._stack:
            inputs = layer(inputs, padding_mask, training, causal=causal)
        outputs = self._layernorm(inputs)
        return outputs
