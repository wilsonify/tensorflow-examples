# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:34:43 2016

@author: tomhope
"""
from __future__ import print_function

import tensorflow as tf
# Import MINST data
from keras.datasets.mnist import load_data


def main():
    mnist = load_data(path="/tmp/data/")

    # Define some parameters
    element_size = 28
    time_steps = 28
    num_classes = 10
    batch_size = 128
    hidden_layer_size = 128

    # Where to save TensorBoard model summaries
    LOG_DIR = "logs/RNN_with_summaries"

    # Create placeholders for inputs, labels
    _inputs = tf.compat.v1.placeholder(tf.float32,
                                       shape=[None, time_steps, element_size],
                                       name='inputs')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='labels')

    # This helper function taken from official TensorFlow documentation,
    # simply add some ops that take care of logging summaries
    def variable_summaries(var):
        with tf.compat.v1.name_scope('summaries'):
            mean = tf.reduce_mean(input_tensor=var)
            tf.compat.v1.summary.scalar('mean', mean)
            with tf.compat.v1.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
            tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
            tf.compat.v1.summary.histogram('histogram', var)

    # Weights and bias for input and hidden layer
    with tf.compat.v1.name_scope('rnn_weights'):
        with tf.compat.v1.name_scope("W_x"):
            Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
            variable_summaries(Wx)
        with tf.compat.v1.name_scope("W_h"):
            Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
            variable_summaries(Wh)
        with tf.compat.v1.name_scope("Bias"):
            b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
            variable_summaries(b_rnn)

    def rnn_step(previous_hidden_state, x):
        current_hidden_state = tf.tanh(
            tf.matmul(previous_hidden_state, Wh) +
            tf.matmul(x, Wx) + b_rnn)

        return current_hidden_state

    # Processing inputs to work with scan function
    # Current input shape: (batch_size, time_steps, element_size)
    processed_input = tf.transpose(a=_inputs, perm=[1, 0, 2])
    # Current input shape now: (time_steps,batch_size, element_size)

    initial_hidden = tf.zeros([batch_size, hidden_layer_size])
    # Getting all state vectors across time
    all_hidden_states = tf.scan(rnn_step,
                                processed_input,
                                initializer=initial_hidden,
                                name='states')

    # Weights for output layers
    with tf.compat.v1.name_scope('linear_layer_weights') as scope:
        with tf.compat.v1.name_scope("W_linear"):
            Wl = tf.Variable(tf.random.truncated_normal([hidden_layer_size, num_classes],
                                                        mean=0, stddev=.01))
            variable_summaries(Wl)
        with tf.compat.v1.name_scope("Bias_linear"):
            bl = tf.Variable(tf.random.truncated_normal([num_classes],
                                                        mean=0, stddev=.01))
            variable_summaries(bl)

    # Apply linear layer to state vector
    def get_linear_layer(hidden_state):
        return tf.matmul(hidden_state, Wl) + bl

    with tf.compat.v1.name_scope('linear_layer_weights') as scope:
        # Iterate across time, apply linear layer to all RNN outputs
        all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
        # Get Last output -- h_28
        output = all_outputs[-1]
        tf.compat.v1.summary.histogram('outputs', output)

    with tf.compat.v1.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf.stop_gradient(y)))
        tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

    with tf.compat.v1.name_scope('train'):
        # Using RMSPropOptimizer
        train_step = tf.compat.v1.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

    with tf.compat.v1.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(input=y, axis=1), tf.argmax(input=output, axis=1))

        accuracy = (tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))) * 100
        tf.compat.v1.summary.scalar('accuracy', accuracy)

    # Merge all the summaries
    merged = tf.compat.v1.summary.merge_all()

    # Get a small test set
    test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
    test_label = mnist.test.labels[:batch_size]

    with tf.compat.v1.Session() as sess:
        # Write summaries to LOG_DIR -- used by TensorBoard
        train_writer = tf.compat.v1.summary.FileWriter(LOG_DIR + '/train',
                                                       graph=tf.compat.v1.get_default_graph())
        test_writer = tf.compat.v1.summary.FileWriter(LOG_DIR + '/test',
                                                      graph=tf.compat.v1.get_default_graph())

        sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(10000):

            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 sequences of 28 pixels
            batch_x = batch_x.reshape((batch_size, time_steps, element_size))
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={_inputs: batch_x, y: batch_y})
            # Add to summaries
            train_writer.add_summary(summary, i)

            if i % 1000 == 0:
                acc, loss, = sess.run([accuracy, cross_entropy],
                                      feed_dict={_inputs: batch_x,
                                                 y: batch_y})
                print("Iter " + str(i) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))
            if i % 100 == 0:
                # Calculate accuracy for 128 mnist test images and
                # add to summaries
                summary, acc = sess.run([merged, accuracy],
                                        feed_dict={_inputs: test_data,
                                                   y: test_label})
                test_writer.add_summary(summary, i)

        test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                                                 y: test_label})
        print("Test Accuracy:", test_acc)


if __name__ == "__main__":
    main()
