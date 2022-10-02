# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:34:43 2016

@author: tomhope
"""
from __future__ import print_function

import pytest
import tensorflow as tf

from tensorflow_examples.c04_convolutional_neural_networks.mnist_cnn import MnistLoader
from tensorflow_examples.c05_text_and_visualizations.BasicRNNCell import get_linear_layer

element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128


@pytest.fixture(name='mnist')
def mnist_fixture():
    mnist = MnistLoader()
    mnist.load_data()
    return mnist


def test_main(mnist):
    tf.compat.v1.disable_eager_execution()
    _inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, time_steps, element_size], name='inputs')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='inputs')
    rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_layer_size)
    outputs, _ = tf.compat.v1.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)
    last_rnn_output = outputs[:, -1, :]
    final_output = get_linear_layer(last_rnn_output)
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=tf.stop_gradient(y))
    cross_entropy = tf.reduce_mean(input_tensor=softmax)
    train_step = tf.compat.v1.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(input=y, axis=1), tf.argmax(input=final_output, axis=1))
    accuracy = (tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))) * 100
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    test_data = mnist.test_images[:batch_size].reshape((-1, time_steps, element_size))
    test_label = mnist.test_labels[:batch_size]

    for i in range(1, 1001):
        batch_x, batch_y = mnist.next_batch(batch_size)
        batch_x = batch_x.reshape((-1, time_steps, element_size))
        sess.run(train_step, feed_dict={_inputs: batch_x, y: batch_y})
        if i % 1000 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: batch_x, y: batch_y})
            loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x, y: batch_y})
            print(f"""Iter {i} Minibatch Loss= {loss}, Training Accuracy= {acc}""")

    testing_accuracy = sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label})
    print(f"Testing Accuracy:{testing_accuracy}")
