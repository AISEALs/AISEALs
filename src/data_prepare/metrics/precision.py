import numpy as np

import tensorflow as tf
from tensorflow.python import debug as tf_debug

def test_tf_precision():
    labels = np.array([[1, 1, 1, 0],
                       [1, 1, 1, 0],
                       [1, 1, 1, 0],
                       [1, 1, 1, 0]], dtype=np.uint8)

    predictions = np.array([[1, 0, 0, 0],
                            [1, 1, 0, 0],
                            [1, 1, 1, 0],
                            [0, 1, 1, 1]], dtype=np.uint8)

    n_batches = len(labels)

    positive_num = (predictions > 0).sum()
    true_positive_num = (labels*predictions > 0).sum()

    print("precision: %1.4f" % (true_positive_num/positive_num))

    labels_ = tf.placeholder(tf.int64, [None])
    predictions_ = tf.placeholder(tf.float32, [None])

    precision, update_op = tf.metrics.precision(labels_, predictions_, name="my_metric")

    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(tf.global_variables_initializer())
    sess.run(running_vars_initializer)

    for i in range(n_batches):
        # Update the running variables on new batch of samples
        feed_dict = {labels_: labels[i], predictions_: predictions[i]}
        sess.run(update_op, feed_dict=feed_dict)

        # Calculate the score
    score = sess.run(precision)
    print("[TF] SCORE: %1.4f" % score)

if __name__ == '__main__':
    test_tf_precision()


