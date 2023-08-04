# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Binary for trianing a RNN-based classifier for the Quick, Draw! data.

python train_lr_model.py \
  --training_data train_file \
  --eval_data eval_data \
  --model_dir /tmp/quickdraw_model/ \
  --cell_type cudnn_lstm

When running on GPUs using --cell_type cudnn_lstm is much faster.

The expected performance is ~75% in 1.5M steps with the default configuration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
import pprint
from data_prepare import youliao_helpers
from text_cnn_model.text_cnn import TextCNN
from common_tools import estimator_util

import tensorflow as tf
from tensorflow.python import debug as tf_debug

tf.flags.DEFINE_string("train_file", "../data/text_classification/tfrecord/training.tfrecord-*", "Data source for the train data.")
tf.flags.DEFINE_string("classes_file", "../data/text_classification/tfrecord/training.tfrecord.classes", "Data source for the train data.")
tf.flags.DEFINE_string("eval_file", "../data/text_classification/tfrecord/eval.tfrecord-*", "Data source for the dev data.")
tf.flags.DEFINE_string("model_dir", "runs/", "Data source for the dev data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate used for training.")

# Training parameters
tf.flags.DEFINE_integer("train_batch_size", 64, "Train Batch Size (default: 64)")
tf.flags.DEFINE_integer("eval_batch_size", 256, "Eval Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_steps", 20000, "Number of max steps (default: 2w)")
tf.flags.DEFINE_integer("eval_steps", 10, "Number of eval steps (default: 10)")
tf.flags.DEFINE_integer("local_eval_frequency", 100, "evaluate once every n train steps")
tf.flags.DEFINE_integer("log_steps", 10, "log by hook every n steps")

tf.flags.DEFINE_integer("checkpoint_steps", 1000, "Save models after this many steps (default: 100)")
tf.flags.DEFINE_integer("save_summary_steps", 0, "Since we provide a SummarySaverHook, we need to disable default SummarySaverHook. To do that we set save_summaries_steps to 0.(default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
# Misc Parameters
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("debug", False, "DEBUG mode")
tf.flags.DEFINE_boolean("tensorboard_debug_address", False, "")

FLAGS = tf.flags.FLAGS

# vocab = learn.preprocessing.VocabularyProcessor.restore('vocab.pickle')
# vocab_size=len(vocab.vocabulary_),
# print("vocab size" + str(vocab_size))
# del vocab
vocab_size = 426343
def get_num_classes():
    with tf.gfile.GFile(FLAGS.classes_file, "r") as f:
        classes = [x for x in f]
    num_classes = len(classes)
    return num_classes

def model_fn(features, labels, mode, params):
    """Model function for TextCNN classifier.
    Args:
      features: dictionary with keys: inks, lengths.
      labels: one hot encoded classes
      mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
      params: a parameter dictionary with the following keys: num_layers,
        num_nodes, batch_size, num_conv, conv_len, num_classes, learning_rate.

    Returns:
      ModelFnOps for Estimator API.
    """
    def _get_input_tensors(features, labels):
        if labels is not None:
            labels = tf.one_hot(tf.reshape(labels, [-1]), params.num_classes)
            # labels = tf.squeeze(tf.one_hot(labels, params.num_classes))
        return features, labels

    features_, labels_ = _get_input_tensors(features, labels)

    dropout_keep_prob = 1.0 if mode == 'eval' else params.dropout_keep_prob
    cnn = TextCNN(
        features=features_,
        labels=labels_,
        sequence_length=features_.shape[1],
        num_classes=params.num_classes,
        vocab_size=params.vocab_size,
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        dropout_keep_prob=dropout_keep_prob,
        l2_reg_lambda=FLAGS.l2_reg_lambda)

    # Decay the learning rate exponentially based on the number of steps.
    # lr = 0.0002
    lr = tf.train.exponential_decay()

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, tf.train.get_global_step())

    training_hooks, training_chief_hooks, evaluation_hooks = create_hooks(cnn, mode, params, lr)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"logits": cnn.scores, "predictions": cnn.y_pred},
        loss=cnn.loss,
        train_op=train_op,
        # eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, cnn.predictions)},
        training_hooks=training_hooks,
        training_chief_hooks=training_chief_hooks,
        evaluation_hooks=evaluation_hooks
    )

def create_hooks(cnn, mode, params, lr):
    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': tf.train.get_global_step(),
                 'loss': cnn.loss,
                 # 'predictions': cnn.predictions,
                 # 'label': tf.argmax(cnn.input_y, 1),
                 'accuracy': cnn.accuracy,
                 'mode': tf.constant(mode)},
        every_n_iter=FLAGS.log_steps)

    hparam_str = "keep_prob:%.3f,l2_reg:%.3f,lr:%.4f" % (FLAGS.dropout_keep_prob, FLAGS.l2_reg_lambda, lr)
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
    lr_summary = tf.summary.scalar('learning_rate', lr)
    summary_dir = os.path.join(params.model_dir, "summaries", mode, hparam_str)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        summary_op=tf.summary.merge([loss_summary, acc_summary, lr_summary]),
        summary_writer=summary_writer
    )

    debug_hook = None
    if FLAGS.debug and FLAGS.tensorboard_debug_address:
        raise ValueError("The --debug and --tensorboard_debug_address flags are mutually exclusive.")
    if FLAGS.debug:
        debug_hook = tf_debug.LocalCLIDebugHook()
    elif FLAGS.tensorboard_debug_address:
        debug_hook = tf_debug.TensorBoardDebugHook(FLAGS.tensorboard_debug_address)

    training_hooks = [logging_hook]
    training_chief_hooks = [summary_hook]
    evaluation_hooks = [logging_hook, summary_hook]
    if debug_hook:
        training_hooks += [debug_hook]
        evaluation_hooks += [debug_hook]
    return training_hooks, training_chief_hooks, evaluation_hooks


def create_estimator(run_config):
    """Creates an Experiment configuration based on the estimator and input fn."""
    model_params = tf.contrib.training.HParams(
        num_classes=get_num_classes(),
        vocab_size=vocab_size,
        dropout_keep_prob=FLAGS.dropout_keep_prob,
        model_dir=run_config.model_dir)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    return estimator

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    timestamp = str(int(time.time()))
    model_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(model_dir))

    attrs = dict([(k, FLAGS[k].value) for k in FLAGS])
    pp = pprint.PrettyPrinter().pprint
    pp(attrs)

    if FLAGS.local_eval_frequency < FLAGS.checkpoint_steps:
        print("local_eval_frequency must be >= checkpoint_steps because evaluate dependent the latest checkpoint!!")
    estimator = create_estimator(
        run_config=tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=FLAGS.checkpoint_steps,
            save_summary_steps=FLAGS.save_summary_steps,
            keep_checkpoint_max=FLAGS.num_checkpoints))

    train_input_fn = youliao_helpers.get_input_fn(mode=tf.estimator.ModeKeys.TRAIN, tfrecord_pattern=FLAGS.train_file, batch_size=FLAGS.train_batch_size)

    eval_input_fn = youliao_helpers.get_input_fn(mode=tf.estimator.ModeKeys.EVAL, tfrecord_pattern=FLAGS.eval_file, batch_size=FLAGS.eval_batch_size)

    for values in estimator_util.continuous_train_and_eval(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            local_eval_frequency=None,
            train_steps=FLAGS.max_steps,
            eval_steps=None):
        # continuous_train_and_eval() yields evaluation metrics after each
        # training epoch. We don't do anything here.
        tf.logging.info("eval result:" + str(values))


if __name__ == "__main__":
    tf.app.run()
