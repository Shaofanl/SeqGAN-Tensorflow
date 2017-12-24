import tensorflow as tf
from SeqGAN.common import highway


# following https://github.com/LantaoYu/SeqGAN/blob/master/discriminator.py
class Discriminator(object):
    def __init__(
            self, seq_len, vocab_size,
            emb_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        input_x = tf.placeholder(
            tf.int32, [None, seq_len], name="input_x")
        input_y = tf.placeholder(
            tf.float32, [None, 2], name="input_y")
        dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):

            # Embedding layer
            emb = tf.Variable(
                tf.random_uniform([vocab_size, emb_size], -1.0, 1.0),
                name="W")
            emb_x = tf.nn.embedding_lookup(emb, input_x)
            emb_x_expand = tf.expand_dims(emb_x, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emb_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]),
                                    name="b")
                    conv = tf.nn.conv2d(
                        emb_x_expand,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, seq_len-filter_size+1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                h_highway = highway(h_pool_flat,
                                    h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_highway, dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal(
                    [num_filters_total, 2], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
                predictions = tf.argmax(scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=scores, labels=input_y)
                loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            with tf.name_scope("accuracy"):
                acc = tf.reduce_mean(tf.to_float(
                    tf.equal(predictions, tf.argmax(input_y, 1))))

        d_opt = tf.train.AdamOptimizer(1e-4)
        train_op = tf.contrib.slim.learning.create_train_op(
            loss, d_opt, aggregation_method=2)

        self.d_summary = tf.summary.merge([
            tf.summary.scalar("d_loss", loss),
            tf.summary.scalar("d_acc", acc),
        ])

        self.input_x = input_x
        self.input_y = input_y
        self.dropout_keep_prob = dropout_keep_prob
        self.train_op = train_op
        self.predictions = predictions

    def predict(self, sess, input_x):
        feed_dict = {self.input_x: input_x,
                     self.dropout_keep_prob: 1.0}
        return sess.run(self.predictions, feed_dict=feed_dict)

    def train(self, sess, input_x, input_y, dropout_keep_prob=0.75):
        feed_dict = {self.input_x: input_x,
                     self.input_y: input_y,
                     self.dropout_keep_prob: dropout_keep_prob}
        _, summary = sess.run([self.train_op, self.d_summary],
                              feed_dict=feed_dict)
        return summary
