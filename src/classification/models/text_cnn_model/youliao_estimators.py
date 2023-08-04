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
import pprint
import os
import sys
sys.path.append(os.getcwd())    # 在命令行中，把当前路径加入到sys.path中
from data_prepare import youliao_helpers
from text_cnn import TextCNN
from data_processor.processor_manager import get_processor
import tensorflow as tf
# from tensorflow.python import debug as tf_debug


tf.flags.DEFINE_string("output_dir", "/Users/jiananliu/AISEALs/models/text_classification/runs/", "Data source for the dev data.")

tf.flags.DEFINE_string("data_dir", "/Users/jiananliu/AISEALs/data/text_classification/", "base data dir")

tf.flags.DEFINE_string("task_name", "tribe_labels", "task name")

tf.flags.DEFINE_string("task_id", "0", "task id")

tf.flags.DEFINE_string("init_checkpoint", "/Users/jiananliu/AISEALs/models/text_classification/runs/models.ckpt", "")

tf.flags.DEFINE_boolean("export_model", False, "")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate used for training.")

# Training parameters
tf.flags.DEFINE_integer("train_batch_size", 64, "Train Batch Size (default: 64)")
tf.flags.DEFINE_integer("eval_batch_size", 256, "Eval Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_steps", 300, "Number of max steps (default: 2w)")
tf.flags.DEFINE_integer("eval_steps", 0, "Number of max steps (default: 2w)")
tf.flags.DEFINE_integer("checkpoint_steps", 100, "Save models after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


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
    def _get_input_tensors(features_, labels_):
        if mode == tf.estimator.ModeKeys.PREDICT:
            features = features_['feature']
        else:
            features = features_
        return features, labels_

    features, labels = _get_input_tensors(features, labels)
    # print(features.shape, labels.shape)
    print(features, labels, mode)
    cnn = TextCNN(
        features=features,
        labels=labels,
        sequence_length=features.shape[1],
        num_classes=params.num_classes,
        vocab_size=params.vocab_size,
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        dropout_keep_prob=params.dropout_keep_prob,
        l2_reg_lambda=FLAGS.l2_reg_lambda,
        mode=mode)

    # tvars = tf.trainable_variables()
    #
    # initialized_variable_names = None
    # init_checkpoint = FLAGS.init_checkpoint
    # if init_checkpoint:
    #     from common_tools import estimator_util
    #     (assignment_map, initialized_variable_names
    #      ) = estimator_util.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    #     print(assignment_map)
    #     print(initialized_variable_names)
    #     tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    #
    # tf.logging.info("**** Trainable Variables ****")
    # for var in tvars:
    #     init_string = ""
    #     if var.name in initialized_variable_names:
    #         init_string = ", *INIT_FROM_CKPT*"
    #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                     init_string)

    if mode == tf.estimator.ModeKeys.TRAIN:
        print("mode loss: {}".format(mode, cnn.loss))
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, tf.train.get_global_step())
        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': tf.train.get_global_step(),
                     'loss': cnn.loss,
                     'predictions': cnn.predictions,
                     'labels': tf.reshape(labels, [-1]),
                     'accuracy': cnn.accuracy},
            every_n_iter=10)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cnn.loss,
            train_op=train_op,
            training_hooks=[logging_hook]
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        print("mode loss: {}".format(mode, cnn.loss))
        logging_hook2 = tf.train.LoggingTensorHook(
            tensors={'step': tf.train.get_global_step(),
                     'loss': cnn.loss,
                     'predictions': cnn.predictions,
                     'labels': tf.reshape(labels, [-1]),
                     'accuracy': cnn.accuracy},
            every_n_iter=10)

        ## todo: 统计各个类别准确率
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cnn.loss,
            eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, cnn.predictions),
                             "mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(labels=tf.reshape(labels, [-1]), predictions=cnn.predictions, num_classes=params.num_classes)},
            evaluation_hooks=[logging_hook2]
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        print("mode: {}".format(mode))
        export_outputs = {
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
                tf.estimator.export.PredictOutput({"predictions": cnn.predictions, "logits": cnn.scores})
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"logits": cnn.scores, "predictions": cnn.predictions},
            export_outputs=export_outputs
        )


def create_estimator_and_specs(run_config, processor):
    """Creates an Experiment configuration based on the estimator and input fn."""
    model_params = tf.contrib.training.HParams(
        num_classes=len(processor.get_labels()),
        vocab_size=len(processor.get_vocab().vocabulary_),
        init_checkpoint=FLAGS.init_checkpoint, #后面没有用
        dropout_keep_prob=FLAGS.dropout_keep_prob)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    # debug_hooks = tf_debug.LocalCLIDebugHook()
    # debug_hooks = None

    train_pattern = os.path.join(processor.tfrecord_dir, "train.tfrecord*")
    train_spec = tf.estimator.TrainSpec(
        input_fn=youliao_helpers.get_input_fn(
            mode=tf.estimator.ModeKeys.TRAIN,
            tfrecord_pattern=train_pattern,
            batch_size=FLAGS.train_batch_size),
        max_steps=FLAGS.max_steps,
        hooks=[]
    )

    eval_steps = None if FLAGS.eval_steps <= 0 else FLAGS.eval_steps
    eval_pattern = os.path.join(processor.tfrecord_dir, "eval.tfrecord*")
    eval_spec = tf.estimator.EvalSpec(
        input_fn=youliao_helpers.get_input_fn(
            mode=tf.estimator.ModeKeys.EVAL,
            tfrecord_pattern=eval_pattern,
            batch_size=FLAGS.eval_batch_size),
        steps=eval_steps,
        hooks=[]
    )

    return estimator, train_spec, eval_spec

def serving_input_receiver_fn():
    seq_length = 512
    """An input receiver that expects a serialized tf.Example."""

    # inputs = {
    #     'feature': tf.placeholder(tf.int64, shape=[seq_length])
    #     # 'label': tf.placeholder(tf.int64, shape=[None])
    # }
    # return tf.estimator.export.build_raw_serving_input_receiver_fn(features=inputs)
    name_to_features = {
        'feature': tf.FixedLenFeature([seq_length], dtype=tf.int64)
    }
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(name_to_features)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    attrs = dict([(k, FLAGS[k].value) for k in FLAGS])
    pp = pprint.PrettyPrinter().pprint
    pp(attrs)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = get_processor(base_dir=FLAGS.data_dir, task_name=FLAGS.task_name, task_id=FLAGS.task_id, use_hdfs=False)

    estimator, train_spec, eval_spec = create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.checkpoint_steps,
            save_summary_steps=100,
            keep_checkpoint_max=FLAGS.num_checkpoints),
        processor=processor)

    if not FLAGS.export_model:
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.export_model:
        export_dir_base = FLAGS.output_dir
        estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn())

    return 0

if __name__ == "__main__":
    tf.app.run()
