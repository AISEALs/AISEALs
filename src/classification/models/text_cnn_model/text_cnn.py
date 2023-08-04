import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda=0.0, features=None, labels=None, mode=None):

        self.num_classes = num_classes
        # Placeholders for input, output and dropout
        self.input_x = features
        # self.input_y = labels
        if mode != tf.estimator.ModeKeys.PREDICT:
            self.input_y = tf.one_hot(tf.reshape(labels, [-1]), num_classes)
        # self.input_y = tf.squeeze(tf.one_hot(y, num_classes))
        self.dropout_keep_prob = dropout_keep_prob

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # with tf.device('/gpu:0'), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):
            #W: [vocab_size, 128]
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            #input_x: [batch_size, 512]
            #embedded_chars: [batch_size, 512, 128]
            #embedded_chars_expanded: [batch_size, 512, 128, 1]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        #filter_sizes: [3,4,5], num_filters:128
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                #conv: [batch_size, 512-filter_size+1, 128-embedding_size+1, num_filters]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                #pooled: [batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        #h_pool: [batch_size, 1, 1, num_filters_total]
        self.h_pool = tf.concat(pooled_outputs, 3)
        #h_pool_flat: [batch_size, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        if mode != tf.estimator.ModeKeys.PREDICT:
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                # losses = tf.reduce_mean(-tf.reduce_sum(self.input_y * tf.log(tf.nn.softmax(self.scores))))
                # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=tf.squeeze(self.input_y))
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, labels)
                # correct_predictions = tf.equal(self.predictions, self.input_y)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

