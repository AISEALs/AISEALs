#/tmp/AISEALs/text_classification/ coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.getcwd())    # 在命令行中，把当前路径加入到sys.path中
print(sys.path)
import modeling
import optimization
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.training import HParams
from data_processor.processor_manager import get_processor


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "base_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_bool(
    "etl_rawdata", False,
    "raw data need etl?")

flags.DEFINE_bool(
    "etl_tfrecord", False,
    "raw data need etl?")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT models. "
    "This specifies the models architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("task_id", "1", "The id of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT models was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the models checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT models).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the models in inference mode on the test set.")

flags.DEFINE_bool("export_model", False, "Whether to export the models with .pb format.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

tf.flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout keep probability (default: 0.5)")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the models checkpoint.")

tf.flags.DEFINE_integer("log_steps", 100, "log by hook every n steps")

# flags.DEFINE_integer("iterations_per_loop", 1000,
#                      "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_boolean("debug", False, "DEBUG mode")
tf.flags.DEFINE_boolean("tensorboard_debug_address", False, "")

flags.DEFINE_boolean("multiple", False, "是否是多分类模型")
flags.DEFINE_string("label_name", "心情_情绪_想法表达", "二分类模型，label_name + 其他")


def file_based_input_fn_builder(tfrecord_pattern, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    # for name in list(example.keys()):
    #   t = example[name]
    #   if t.dtype == tf.int64:
    #     t = tf.to_int32(t)
    #   example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    # batch_size = params["batch_size"]
    if is_training:
        batch_size = params.train_batch_size
    else:
        batch_size = params.eval_batch_size

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
    d = d.interleave(tf.data.TFRecordDataset, cycle_length=10, block_length=16)

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    # from tensorflow.contrib.data import map_and_batch
    d = d.apply(
        # tf.contrib.data.map_and_batch(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            num_parallel_calls=5))
    d = d.prefetch(5)
    # d = d.make_one_shot_iterator().get_next()

    return d

  return input_fn

def serving_input_receiver_fn():
  seq_length = FLAGS.max_seq_length
  """An input receiver that expects a serialized tf.Example."""

  # inputs = {
  #     'input_ids': tf.placeholder(tf.int64, shape=[None, seq_length]),
  #     'input_mask': tf.placeholder(tf.int64, shape=[None, seq_length]),
  #     'segment_ids': tf.placeholder(tf.int64, shape=[None, seq_length]),
  #     'label_ids': tf.placeholder(tf.int32, shape=[None])
  # }
  # return tf.estimator.export.build_raw_serving_input_receiver_fn(features=inputs)
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  return tf.estimator.export.build_parsing_serving_input_receiver_fn(name_to_features)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification models."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use models.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=FLAGS.dropout_keep_prob)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

def create_model_multi(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification models."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use models.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    output_weights_sigmoid = tf.get_variable(
        "output_weights_sigmoid", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias_sigmoid = tf.get_variable(
        "output_bias_sigmoid", [num_labels], initializer=tf.zeros_initializer())

    if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=FLAGS.dropout_keep_prob)

    with tf.variable_scope("tag_loss"):
        logits = tf.matmul(output_layer, output_weights_sigmoid, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias_sigmoid)
        tag_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    with tf.variable_scope("cate_loss"):
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        cate_loss = tf.reduce_mean(per_example_loss)

    loss = tag_loss + cate_loss
    return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)


    # tf.logging.info(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

    # ModeKeys.PREDICT
    # predictions = {
    #     'class_ids': predicted_classes[:, tf.newaxis],
    #     'probabilities': probabilities,
    #     'logits': logits,
    # }
    predictions = tf.argmax(logits, -1, output_type=tf.int64)
    correct_predictions = tf.equal(predictions, label_ids)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    # accuracy = tf.metrics.accuracy(label_ids, predictions)

    training_hooks, training_chief_hooks, evaluation_hooks = create_hooks(mode, total_loss, predictions, accuracy, params, learning_rate)

    def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

    export_outputs = {
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
            tf.estimator.export.PredictOutput({"predictions": predictions, "logits": logits})
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=metric_fn(per_example_loss, label_ids, logits),
        training_hooks=training_hooks,
        training_chief_hooks=training_chief_hooks,
        evaluation_hooks=evaluation_hooks,
        export_outputs=export_outputs
    )

  return model_fn

def create_hooks(mode, loss, predictions, accuracy, params, lr):
    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': tf.train.get_global_step(),
                 'loss': loss,
                 'predictions': predictions,
                 'accuracy': accuracy,
                 'mode': tf.constant(mode)},
        every_n_iter=FLAGS.log_steps)

    hparam_str = "keep_prob:%.3f,lr:%.4f" % (FLAGS.dropout_keep_prob, lr)
    loss_summary = tf.summary.scalar("loss", loss)
    acc_summary = tf.summary.scalar("accuracy", accuracy)
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


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT models "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  processor = get_processor(FLAGS.base_dir, task_name, FLAGS.task_id, use_hdfs=False, multiple=FLAGS.multiple, label_name=FLAGS.label_name)

  label_list = processor.label_mapping_list()
  print("labels: {} size: {}".format(str(label_list), len(label_list)))

  run_config=tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      save_summary_steps=0,
      keep_checkpoint_max=1)


  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    num_train_steps = int(
        processor.get_train_num() / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  hParams = HParams(
      model_dir=run_config.model_dir,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params=hParams)
  #1.params will pass to model_fn_builder.model_fn(..., params).
  #2.params will pass to estimator.train(input_fn=...) input_fn(params).

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", processor.get_train_num())
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_pattern = os.path.join(processor.tfrecord_dir, "train_*.tfrecord")
    train_input_fn = file_based_input_fn_builder(
        tfrecord_pattern=train_pattern,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", processor.get_eval_num())
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_pattern = os.path.join(processor.tfrecord_dir, "eval_*.tfrecord")
    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(
        tfrecord_pattern=eval_pattern,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=None)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.base_dir)
    predict_pattern = os.path.join(processor.tfrecord_dir, "predict.tfrecord")

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
      tfrecord_pattern=predict_pattern,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      tf.logging.info("***** Predict results *****")
      for prediction in result:
        output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
        writer.write(output_line)

  if FLAGS.export_model:
    export_dir_base = FLAGS.output_dir
    estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn())

def test():
  train_pattern = os.path.join("/Users/jiananliu/AISEALs/data/text_classification/youliao_raw_data/tfrecord", "eval.tfrecord*")
  params = HParams(
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)
  dataset = file_based_input_fn_builder(
      tfrecord_pattern=train_pattern,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)(params=params)

  with tf.Session() as sess:
      # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      while True:
          try:
              x = sess.run([dataset])
              # print(x[0][1].reshape(1, 8))
              print(x)
          except tf.errors.OutOfRangeError:
              break


if __name__ == "__main__":
  flags.mark_flag_as_required("base_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
  # test()
