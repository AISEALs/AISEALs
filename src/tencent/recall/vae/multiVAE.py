import numerous
import tensorflow as tf
import tensorflow.keras as K
from model_zoo.layers import MLP
import numerous
import numpy as np
from typing import List, Optional, Tuple
from tensorflow.python.keras import backend as PK
from tensorflow.python.ops import rnn_cell
from numerous.optimizers.adam import Adam
from numerous.distributions.uniform import Uniform
from numerous.distributions.normal import Normal
from numerous.framework.dense_parameter import DenseParameter

if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

from model_zoo.inputs import Inputs


class Inputs_user_define(object):
    def __init__(self):
        print("[DEBUG] init rank inputs")

    def build_numerous_inputs(self):
        # TODO: 特征解析处理，返回类型是字典，其中key为特征名称，value为特征对应的tensor值
        inputs = Inputs()
        # Add spare features.
        sparse_slots = [297, 287]
        dense_slots = [30015, 30017]
        sparse_feature_dim = 512
        session_length = 50

        inputs.add_dense_feature(
            feature_name="item_ids",
            slot_id=30015,
            lower_bound=1,
            upper_bound=session_length
        )

        inputs.add_dense_feature(
            feature_name="item_ids_weight",
            slot_id=30017,
            lower_bound=1,
            upper_bound=session_length
        )

        inputs.add_varlen_features(
            feature_names=["%s_ids" % str(slotid) for slotid in sparse_slots],
            slot_ids=sparse_slots,
            embedding_dims=[sparse_feature_dim] * len(sparse_slots),
            lengths=[session_length, session_length])

        features = inputs.build()

        features["mode"] = "train"
        return features


class MultiVAE(tf.keras.layers.Layer):
    """Partially-regularized variational autoencoder with multinomial likelihood.
    """

    def __init__(self,
                 decoder_dims: List[int],
                 encoder_dims: Optional[List[int]] = None,
                 l2_reg: int = 0.01,
                 input_mode: str = "sparse"):
        super().__init__()

        self.decoder_dims = decoder_dims
        if encoder_dims is None:
            self.encoder_dims = decoder_dims[::-1]
        else:
            if encoder_dims[0] != decoder_dims[-1]:
                raise ValueError("Input and output dimension must equal each other.")
            if encoder_dims[-1] != decoder_dims[0]:
                raise ValueError("Latent dimension for p- and q-network mismatches.")
            self.encoder_dims = encoder_dims

        if input_mode == "sparse":
            self.encoder_dims = self.encoder_dims[1:]

        self.dims = self.encoder_dims + self.decoder_dims[1:]

        self.l2_reg = l2_reg
        self.layers_encoder, self.layers_decoder = [], []
        self._construct_weights()

    def _construct_weights(self):
        for i, (_, d_out) in enumerate(zip(self.encoder_dims[:-1], self.encoder_dims[1:])):
            if i == len(self.encoder_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2

            dense_layer = tf.keras.layers.Dense(
                units=d_out,
                activation=None,
                kernel_initializer="glorot_normal",
                bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001)
            )
            self.layers_encoder.append(dense_layer)

        for _, (_, d_out) in enumerate(zip(self.decoder_dims[:-1], self.decoder_dims[1:])):
            dense_layer = tf.keras.layers.Dense(
                units=d_out,
                activation=None,
                kernel_initializer="glorot_normal",
                bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001)
            )
            self.layers_decoder.append(dense_layer)

    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor],
             **kwargs) -> tf.Tensor:  # pylint: disable=arguments-differ
        """Invoke the MultiVAE module.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): \
                Bag-of-words click vector from user u.
            and numeorus is_training placeholder.

        Returns:
            Union[tf.Tensor, tf.Tensor]: logits and Kl divergence.
        """
        x_input, is_training = inputs
        mu_encoder, std_encoder, kl_divergence = self.encoder_fn(x_input)
        epsilon = tf.random.normal(tf.shape(std_encoder))
        sample_z = mu_encoder + is_training * epsilon * std_encoder
        logits = self.decoder_fn(sample_z)

        return logits, kl_divergence

    def encoder_fn(self, x_input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Encoder subgraph.
        """
        mu_encoder, std_encoder, kl_divergence = None, None, None

        hidden = tf.nn.l2_normalize(x_input, axis=1)

        for i, layer in enumerate(self.layers_encoder):
            hidden = layer(hidden)
            if i != len(self.layers_encoder) - 1:
                hidden = tf.nn.tanh(hidden)
            else:
                mu_encoder = hidden[:, :self.encoder_dims[-1]]
                logvar_q = hidden[:, self.encoder_dims[-1]:]

                std_encoder = tf.exp(0.5 * logvar_q)
                kl_divergence = tf.reduce_mean(tf.reduce_sum(
                    0.5 * (-logvar_q + tf.exp(logvar_q) + mu_encoder ** 2 - 1), axis=1))
        return mu_encoder, std_encoder, kl_divergence

    def decoder_fn(self, z_input: tf.Tensor) -> tf.Tensor:
        """Decoder subgraph.
        """
        hidden = z_input
        for i, layer in enumerate(self.layers_decoder):
            hidden = layer(hidden)

            if i != len(self.layers_decoder) - 1:
                hidden = tf.nn.tanh(hidden)

            if i == len(self.layers_decoder) - 2:
                one_bias = tf.ones(shape=[tf.shape(hidden)[0], 1])
                self.user_vector = tf.identity(
                    tf.concat([hidden, one_bias], axis=1))

        return hidden


class Model_user_define():
    def __init__(self, inputs):
        self.inputs = inputs
        print("<<<<<<<<<<<<< estimator members >>>>>>>>>>>>>>>")
        print(self.__dict__.items())
        print("============= estimator members ===============")
        self.multi_vae = MultiVAE(
            decoder_dims=[256, 512, 100000],
            encoder_dims=None,
            input_mode="sparse"
        )

    def _multi_hot_encode(self,
                          session: tf.Tensor,
                          weight: tf.Tensor,
                          item_num: int):
        """Convert the user session & weight sequence to multi-hot vectors.

        Args:
            session (tf.Tensor): User session with the shape of (batch, session_length).
            weight (tf.Tensor): item's weight of session  (batch, session_length).
            item_num (int): Total number of the items.

        Returns:
            tf.Tensor: A multi-hot session tensor with shape of (batch, item_num).
            tf.Tensor: A multi-hot weight tensor with shape of (batch, item_num).
        """
        session = tf.cast(session, dtype=tf.int32)
        session = tf.one_hot(session, depth=item_num, dtype=tf.int32)

        if weight is not None:
            session = tf.cast(session, dtype=tf.float32)
            weight = tf.expand_dims(weight, axis=2)
            session = session * weight
            weight = tf.reduce_sum(session, axis=1)

        session = tf.reduce_any(tf.greater(session, 0), axis=1)
        session = tf.cast(session, dtype=tf.float32)

        weight = session if weight is None else weight

        return session, weight

    def _negative_log_likelihood_loss(
            self,
            x_weight: tf.Tensor,
            logits: tf.Tensor) -> tf.Tensor:
        """Compute negative log likelihood loss.

        Args:
            x_weight (tf.Tensor): Input weight distribution for the vae.
            logits (tf.Tensor): Output distribution by the decoder.

        Returns:
            tf.Tensor: The finnal loss.
        """
        log_softmax_var = tf.nn.log_softmax(logits)
        # per-user average negative log-likelihood
        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x_weight, axis=1
        ))
        return neg_ll

    def model(self):
        is_training = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='is_training')
        numerous.reader.ControlPlaceholder(is_training, training_phase=1.0, inference_phase=0.0)

        sparse_slots = [297, 287]
        input_list = []
        input_mask_list = []
        item_num = 100000
        anneal = 0.1

        for slotid in sparse_slots:
            _emb = self.inputs["%s_ids" % slotid]
            _mask = self.inputs["%s_ids_mask" % slotid]
            input_list.append(_emb)
            input_mask_list.append(_mask)

        concat_input = tf.concat(input_list, axis=1)
        concat_mask = tf.cast(tf.concat(input_mask_list, axis=1), dtype=tf.float32)

        valid_len = tf.reduce_sum(concat_mask, axis=-1, keepdims=True)
        mean_pool_in = tf.truediv(tf.reduce_sum(concat_input, axis=1), valid_len)  # (batch_size, emb_dim)

        dense_input = tf.concat(self.inputs["item_ids"], axis=1)  # (batch_size, session_length)

        label_weight = None
        label_weight = tf.concat(self.inputs["item_ids_weight"], axis=1)  # (batch_size, session_length)

        mh_in, mh_weight = self._multi_hot_encode(dense_input, label_weight, item_num)

        logits, kl_divergence = self.multi_vae((mean_pool_in, is_training))
        user_vec = tf.identity(self.multi_vae.user_vector, name="user_embedding")

        neg_ll = loss = self._negative_log_likelihood_loss(mh_weight, logits)
        neg_elbo = neg_ll + anneal * kl_divergence

        loss = neg_elbo
        loss = tf.reduce_mean(loss)

        # 无量评价指标
        metrics = []
        metrics.append(numerous.metrics.DistributeMean(values=loss, name='loss'))

        if self.inputs["mode"] == "item_vector":
            item_id = tf.cast(self.inputs["item_ids"], dtype=tf.int32)
            item0_w = tf.nn.embedding_lookup(
                tf.transpose(self.multi_vae.layers_decoder[-1].variables[0]), item_id[:, 0])
            item0_b = tf.nn.embedding_lookup(
                tf.transpose(self.multi_vae.layers_decoder[-1].variables[1]), item_id[:, 0])
            item0 = tf.concat([item0_w, tf.expand_dims(item0_b, axis=1)], axis=1, name="item_embedding")

        return loss, metrics