import tensorflow as tf
from tensorflow.python import debug as tf_debug

labels = tf.placeholder(tf.int64, [1, 3])
predictions = tf.constant([[7, 5, 10, 6, 3, 1, 8, 12, 31, 88]], tf.int64)
precision, update_op = tf.metrics.precision_at_k(labels, predictions, 8)

sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.local_variables_initializer())

stream_vars = [i for i in tf.local_variables()]
#Get the local variables true_positive and false_positive

# print("[PRECSION_1]: ", sess.run(precision, {labels:[[1, 5, 10]]})) # nan
#tf.metrics.precision maintains two variables true_positives
#and  false_positives, each starts at zero.
#so the output at this step is 'nan'

print("[UPDATE_OP_1]:", sess.run(update_op, {labels:[[1, 5, 10]]})) #0.2
#when the update_op is called, it updates true_positives
#and false_positives using labels and predictions.

print("[STREAM_VARS_1]:",sess.run(stream_vars)) #[2.0, 8.0]
# Get true positive rate and false positive rate

print("[PRECISION_1]:", sess.run(precision, {labels:[[1, 10, 15]]})) # 0.2
#So calling precision will use true_positives and false_positives and outputs 0.2

print("[UPDATE_OP_2]:", sess.run(update_op, {labels:[[1, 10, 15]]})) #0.15
#the update_op updates the values to the new calculated value 0.15.

print("[STREAM_VARS_2]:",sess.run(stream_vars)) #[3.0, 17.0]
