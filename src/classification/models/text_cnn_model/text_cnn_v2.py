import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, features=None, labels=None,features_test=None, labels_test=None):

        self.num_classes = num_classes
        # Placeholders for input, output and dropout
        if features == None:
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        else:
            self.input_x = features
            self.input_y = labels
            self.input_x_test=features_test
            self.input_y_test=labels_test

        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob=0.5
        # self.input_x = x
        # self.input_y = tf.squeeze(tf.one_hot(y, num_classes))


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        l2_loss_test = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_test = tf.nn.embedding_lookup(self.W, self.input_x_test)
            self.embedded_chars_expanded_test = tf.expand_dims(self.embedded_chars_test, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        pooled_outputs_test = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                conv_test = tf.nn.conv2d(
                    self.embedded_chars_expanded_test,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_test")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h_test = tf.nn.relu(tf.nn.bias_add(conv_test, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_test = tf.nn.max_pool(
                    h_test,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_test")
                pooled_outputs.append(pooled)
                pooled_outputs_test.append(pooled_test)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_test = tf.concat(pooled_outputs_test, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_pool_flat_test = tf.reshape(self.h_pool_test, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            self.h_drop_test = tf.nn.dropout(self.h_pool_flat_test, 1.0)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss_test += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            l2_loss_test += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores_test = tf.nn.xw_plus_b(self.h_drop_test, W, b, name="scores_test")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.predictions_test = tf.argmax(self.scores_test, 1, name="predictions_test")
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses_test = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_test, labels=self.input_y_test)
            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=tf.squeeze(self.input_y))
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.loss_test = tf.reduce_mean(losses_test) + l2_reg_lambda * l2_loss_test

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            correct_predictions_test = tf.equal(self.predictions_test, tf.argmax(self.input_y_test, 1))
            # correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.accuracy_test = tf.reduce_mean(tf.cast(correct_predictions_test, "float"), name="accuracy_test")

    # def set_feature_label(self, x, y):
    #     self.input_x = x
    #     self.input_y = tf.squeeze(tf.one_hot(y, self.num_classes))
        # tf.assign(self.input_x, x)
        # tf.assign(self.input_y, y)

