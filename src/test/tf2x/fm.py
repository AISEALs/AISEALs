import tensorflow as tf

def fm(inputs):
    assert tf.shape(inputs) == 3
    square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
    sum_of_square = tf.reduce_sum(inputs * inputs, axis=1, keepdims=True)
    return tf.reduce_sum(0.5*(square_of_sum - sum_of_square), axis=2)