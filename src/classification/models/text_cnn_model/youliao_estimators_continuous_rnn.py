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
import warnings
import os
import pprint
from data_prepare import youliao_helpers
from text_cnn_model.text_rnn import TextRNN
from common_tools import estimator_util

import tensorflow as tf


warnings.filterwarnings("ignore")

# Parameters
# ==================================================

# Data loading parameters
tf.flags.DEFINE_string('data_file', './data/traindata', "Data source for the text data")
tf.flags.DEFINE_float('test_size', 0.05, "Percentage of data to use for validation and test (default: 0.05)")
tf.flags.DEFINE_integer('vocab_size', 9000, "Select words to build vocabulary, according to term frequency (default: 9000)")
tf.flags.DEFINE_integer('sequence_length', 512, "Padding sentences to same length, cut off when necessary (default: 100)")

# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 128, "Dimension of word embedding (default: 128)")
tf.flags.DEFINE_integer('rnn_size', 512, "Dimension of rnn layer (default: 100)")
tf.flags.DEFINE_integer('num_layers', 1, "Number of rnn layer (default: 1)")
tf.flags.DEFINE_integer('attention_size', 1024, "Dimension of attention layer (default: 100)")
tf.flags.DEFINE_float('learning_rate', 0.001, "Learning rate for models training (default: 0.001)")
tf.flags.DEFINE_float('grad_clip', 5.0, "Gradients clipping threshold (default: 5.0)")

# Training parameters
tf.flags.DEFINE_integer('batch_size', 512, "Batch size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
tf.flags.DEFINE_string('init_embedding_path', None, "Using pre-trained word embedding, npy file format")
tf.flags.DEFINE_string('init_model_path', None, "Continue training from saved models at this path")
tf.flags.DEFINE_integer('evaluate_every', 50, "Evaluate models on val set after this many steps (default: 50)")

# Tensorflow parameters
tf.flags.DEFINE_boolean('gpu_allow_growth', True, "GPU memory allocation mode (default: True)")



tf.flags.DEFINE_string("train_file", "../data/text_classification/tfrecord/training.tfrecord-*", "Data source for the train data.")
tf.flags.DEFINE_string("classes_file", "../data/text_classification/tfrecord/training.tfrecord.classes", "Data source for the train data.")
tf.flags.DEFINE_string("eval_file", "../data/text_classification/tfrecord/eval.tfrecord-*", "Data source for the dev data.")
tf.flags.DEFINE_string("model_dir", "runs/", "Data source for the dev data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("train_batch_size", 64, "Train Batch Size (default: 64)")
tf.flags.DEFINE_integer("eval_batch_size", 64, "Eval Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_steps", 20000, "Number of max steps (default: 2w)")
tf.flags.DEFINE_integer("eval_steps", 10, "Number of eval steps (default: 10)")
tf.flags.DEFINE_integer("local_eval_frequency", 100, "evaluate once every n train steps")

tf.flags.DEFINE_integer("checkpoint_steps", 100, "Save models after this many steps (default: 100)")
tf.flags.DEFINE_integer("save_summary_steps", 0, "Since we provide a SummarySaverHook, we need to disable default SummarySaverHook. To do that we set save_summaries_steps to 0.(default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# vocab = learn.preprocessing.VocabularyProcessor.restore('vocab.pickle')
# vocab_size=len(vocab.vocabulary_),
# print("vocab size" + str(vocab_size))
# del vocab
vocab_size = 426343
def get_num_classes():
    with tf.gfile.GFile(FLAGS.classes_file, "rb") as f:
        classes = [x for x in f]
    num_classes = len(classes)
    return num_classes

def model_fn(features, labels, mode, params):
    """Model function for TextRNN classifier.
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
        # seqlen = tf.squeeze(tf.slice(features,[0,0],[FLAGS.train_batch_size,0]))
        return features, labels

    features_, labels_ = _get_input_tensors(features, labels)

    dropout_keep_prob = 1.0 if mode == 'eval' else params.dropout_keep_prob
    # fixed_seq_len = tf.constant([512 for x in range(FLAGS.train_batch_size)])
    #fixed_seq_len = np.array([features_.shape[1] for x in range(FLAGS.train_batch_size)])
    #fixed_seq_len = tf.constant([features_.shape[1]])
    rnn = TextRNN(
        vocab_size=params.vocab_size,
        embedding_size=FLAGS.embedding_dim,
        sequence_length = FLAGS.sequence_length,
        rnn_size=FLAGS.rnn_size,
        num_layers=FLAGS.num_layers,
        attention_size=FLAGS.attention_size,
        num_classes=params.num_classes,
        learning_rate=FLAGS.learning_rate,
        grad_clip=FLAGS.grad_clip,
        dropout_keep_prob=dropout_keep_prob,
        batch_size = FLAGS.train_batch_size,
        features=features_,
        labels=labels_
    )

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': rnn.global_step,
                 'loss': rnn.loss,
                 # 'predictions': cnn.predictions,
                 # 'label': tf.argmax(cnn.input_y, 1),
                 'accuracy': rnn.accuracy,
                 'mode': tf.constant(mode)},
        every_n_iter=10)

    loss_summary = tf.summary.scalar("loss", rnn.loss)
    acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)
    summary_dir = os.path.join(params.model_dir, "summaries", mode)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        summary_op=tf.summary.merge([loss_summary, acc_summary]),
        summary_writer=summary_writer
    )

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"logits": rnn.logits, "predictions": rnn.y_pred},
        loss=rnn.loss,
        train_op= rnn.train_op,
        eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, rnn.y_pred)},
        training_hooks=[logging_hook],
        training_chief_hooks=[summary_hook],
        evaluation_hooks=[logging_hook, summary_hook]
    )

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
            eval_steps=FLAGS.eval_steps):
        # continuous_train_and_eval() yields evaluation metrics after each
        # training epoch. We don't do anything here.
        tf.logging.info("eval result:" + str(values))


if __name__ == "__main__":
    tf.app.run()
