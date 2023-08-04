import tensorflow as tf

x_data = tf.placeholder(tf.float32)

dimension = 8
embedding_w = tf.get_variable("sparse_doc_w_0", shape=[1, dimension], dtype=tf.float32)

output = tf.matmul(x_data, embedding_w)

with tf.Session() as sess:
    # sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.print(embedding_w))
    print(sess.run(output, feed_dict={x_data: [[3.], [1.]]}))