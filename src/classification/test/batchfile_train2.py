# -*- coding:utf-8 -*-
'''
@author:zjb
@file:batchfile_train2.py
测试生成tfrecord文件
输入youliao2dataset.py生成的tfrecord文件 *.train  , *.test ,vocabulary文件,label_index文件
'''

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn_model.text_cnn_v2 import TextCNN
from tensorflow.contrib import learn
import functools
import traceback
import sys

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("data_file_dir", "D:\\p_project\\data\\youliao\\dataset/", "Data source for the  data.")
#tf.flags.DEFINE_string("mid_file_dir", "D:\\p_project\\cnn-text-classification-tf-master\\cnn-text-classification-tf-master\\data\\sougoutrain\\", "Data source for the sougou data.")
tf.flags.DEFINE_string("vocab_file", "vocab.v1", "Data source for the vocab.")
tf.flags.DEFINE_string("index_label", "index.label", "Data source for the index label.")

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_length", 512, "sequence_length")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate models on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save models after this many steps (default: 100)")
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

def parser(example_proto,class_num=None):
    """Parse a single record which is expected to be a tensorflow.Example."""
    # features = {"title": tf.FixedLenFeature([], tf.int64), "content": tf.FixedLenFeature([], tf.int64),
    #            "category": tf.FixedLenFeature([], tf.int64)}
    feature_to_type = {
        "title": tf.FixedLenFeature([512], tf.int64),
        "content": tf.FixedLenFeature([512], tf.int64),
        "category": tf.FixedLenFeature([1], tf.int64)
        # 'feature': tf.FixedLenFeature([512], dtype=tf.int64)
    }


    parsed_features = tf.parse_single_example(example_proto, feature_to_type)

    labels = parsed_features['category']
    #labels=tf.reshape(labels)
    labels_onehot = tf.squeeze(tf.one_hot(labels,class_num))

    # parsed_features["feature"] = tf.sparse_tensor_to_dense(parsed_features["feature"])
    return parsed_features['title'], parsed_features['content'], labels_onehot



def train():

    # Training
    # ==================================================
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.data_file_dir+FLAGS.vocab_file)
    label_count=0
    with open(FLAGS.data_file_dir + "index.label", 'r', encoding='UTF-8') as label_f:
        label_count = len(label_f.readlines())



    with tf.Graph().as_default(),tf.device('/GPU:0'):
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            train_files = tf.train.match_filenames_once(FLAGS.data_file_dir + "*.train")
            dataset = tf.data.TFRecordDataset(train_files)
            dataset = dataset.map(functools.partial(parser, class_num=label_count),
                                  num_parallel_calls=10)
            shuffle_buffer = 10000
            dataset = dataset.prefetch(40000).shuffle(shuffle_buffer).batch(FLAGS.batch_size)
            dataset = dataset.repeat(FLAGS.num_epochs)
            iterator = dataset.make_initializable_iterator()
            title_batch, content_batch, label_batch = iterator.get_next()


            test_files = tf.train.match_filenames_once(FLAGS.data_file_dir + "*.test")
            datasettest = tf.data.TFRecordDataset(test_files)
            datasettest = datasettest.map(functools.partial(parser, class_num=label_count),
                                  num_parallel_calls=10)
            batch_size = 1024
            datasettest = datasettest.batch(batch_size)
            iterator_test = datasettest.make_initializable_iterator()
            title_batch_test, content_batch_test, label_batch_test = iterator_test.get_next()



            cnn = TextCNN(
                sequence_length=FLAGS.sequence_length,
                num_classes=label_count,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                features=content_batch, labels=label_batch,
                features_test=content_batch_test, labels_test=label_batch_test
            )

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
            loss_summary_test = tf.summary.scalar("loss_test", cnn.loss_test)
            acc_summary_test = tf.summary.scalar("accuracy_test", cnn.accuracy_test)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary_test, acc_summary_test])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "models")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run((tf.global_variables_initializer(),
                      tf.local_variables_initializer()))

            def train_step():
                """
                A single training step
                """
                feed_dict = {
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                #ttt = sess.run(cnn.input_y)
                #print(ttt.shape)
                _, step, summaries, loss, accuracy= sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy])
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                sys.stdout.flush()
                train_summary_writer.add_summary(summaries, step)

            def dev_step(writer=None):
                """
                Evaluates models on a dev set
                """
                test_results = []
                test_labels = []
                test_loss=[]
                dev_count=0

                sess.run(iterator_test.initializer)
                while True:
                    try:
                        feed_dict = {
                            cnn.dropout_keep_prob: 1.0
                        }

                        t_step, t_loss,t_predictions_test ,t_label= sess.run(
                            [global_step, cnn.loss_test,cnn.predictions_test,cnn.input_y_test])
                        #print(t_loss)
                        #print(t_x.shape)
                        dev_count=dev_count + t_label.shape[0]
                        test_results.extend(t_predictions_test)
                        test_labels.extend(t_label)
                        test_loss.extend([t_loss*t_label.shape[0]])
                    except tf.errors.OutOfRangeError:
                        break

                correct = [float(y == y_) for (y, y_) in zip(test_results, np.argmax(test_labels,1))]
                accuracy = sum(correct) / len(correct)
                dev_loss=sum(test_loss)/dev_count
                #print("Test accuracy is:", accuracy)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, t_step, dev_loss, accuracy))
                sys.stdout.flush()
                #if writer:
                    #writer.add_summary(summaries, step)


            #    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            sess.run(iterator.initializer)
            #t1, t2, t3 = sess.run([title_batch, content_batch, label_batch])
            #print(t3)
            while True:
                try:
                    #x_batch, t2,y_batch = sess.run([title_batch, content, label])
                    #print(x_batch.shape)
                    #print(y_batch.shape)
                    #y_batch=np.eye(label_count)[y_batch.reshape(-1)]
                    train_step()
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step( writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved models checkpoint to {}\n".format(path))

                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    break




def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()