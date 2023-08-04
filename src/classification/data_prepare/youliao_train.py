#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_prepare import youliao_helpers
from text_cnn_model import TextCNN

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data", "data/text_classification/tfrecord/training.tfrecord-*", "Data source for the train data.")
tf.flags.DEFINE_string("train_classes", "data/text_classification/tfrecord/training.tfrecord.classes", "Data source for the train data.")
tf.flags.DEFINE_string("dev_data", "data/text_classification/tfrecord/eval.tfrecord-*", "Data source for the dev data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 3000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate models on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save models after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("max_document_length", "200", "max length of each document")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def read_data():
    x_train, y_train = youliao_helpers.get_input_fn("train", FLAGS.train_data, FLAGS.batch_size)()
    x_eval, y_eval = youliao_helpers.get_input_fn("eval", FLAGS.train_data, FLAGS.batch_size)()
    return x_train, y_train, x_eval, y_eval

def train(x_train, y_train, x_dev, y_dev, vocab_size, num_classes):
    # Training
    # ==================================================

    # with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=num_classes,
            vocab_size=vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "models")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            # cnn.set_feature_label(x_batch, y_batch)
            y_ = tf.squeeze(tf.one_hot(y_batch, num_classes))
            # x, y  = sess.run([x_batch, y_])
            feed_dict = {
                cnn.input_x: x,
                cnn.input_y: y,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            y, prediction, _, step, summaries, loss, accuracy = sess.run(
                [cnn.input_y, cnn.y_pred, train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print('------------')
            print('predict:' + str(prediction))
            print('label  :' + str(np.argmax(y, axis=1)))
            print(np.mean(np.equal(prediction, np.argmax(y, axis=1))))
            print('------------')
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates models on a dev set
            """
            # cnn.set_feature_label(x_batch, y_batch)
            y_ = tf.squeeze(tf.one_hot(y_batch, num_classes))
            x, y  = sess.run([x_batch, y_])
            feed_dict = {
                cnn.input_x: x,
                cnn.input_y: y,
                cnn.dropout_keep_prob: 1.0
            }
            # step, summaries, loss, accuracy = sess.run(
            #     [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
            #     feed_dict)
            y, prediction, step, summaries, loss, accuracy = sess.run([cnn.input_y, cnn.y_pred, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print('------------')
            print('predict:' + str(prediction))
            print('label  :' + str(np.argmax(y, axis=1)))
            print(np.mean(np.equal(prediction, np.argmax(y, axis=1))))
            print('------------')
            if writer:
                writer.add_summary(summaries, step)

        # Training loop. For each batch...
        current_step = 0
        while current_step <= FLAGS.num_epochs:
            train_step(x_train, y_train)
            # current_step = tf.train.global_step(sess, global_step)
            # if current_step % FLAGS.evaluate_every == 0:
            #     print("\nEvaluation:")
            #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
            #     print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved models checkpoint to {}\n".format(path))

def main(argv=None):
    # vocab = learn.preprocessing.VocabularyProcessor.restore('vocab.pickle')
    # vocab_size=len(vocab.vocabulary_),
    # print("vocab size" + str(vocab_size))
    # del vocab
    vocab_size = 426343

    with open(FLAGS.train_classes, "rb") as f:
        num_classes = len(f.readlines())

    x_train, y_train, x_dev, y_dev = read_data()
    train(x_train, y_train, x_dev, y_dev, vocab_size, num_classes)

if __name__ == '__main__':
    tf.app.run()
