# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:34:43 2016

@author: tomhope
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

_inputs = tf.compat.v1.placeholder(tf.float32,
                         shape=[None, time_steps, element_size],
                         name='inputs')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='inputs')

# TensorFlow built-in functions
rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.compat.v1.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

Wl = tf.Variable(tf.random.truncated_normal([hidden_layer_size, num_classes],
                                     mean=0, stddev=.01))
bl = tf.Variable(tf.random.truncated_normal([num_classes], mean=0, stddev=.01))


def get_linear_layer(vector):
    return tf.matmul(vector, Wl) + bl


last_rnn_output = outputs[:, -1, :]
final_output = get_linear_layer(last_rnn_output)

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=tf.stop_gradient(y))
cross_entropy = tf.reduce_mean(input_tensor=softmax)
train_step = tf.compat.v1.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(input=y, axis=1), tf.argmax(input=final_output, axis=1))
accuracy = (tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))) * 100

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

test_data = mnist.test.images[:batch_size].reshape(
    (-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

for i in range(3001):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, time_steps, element_size))
    sess.run(train_step, feed_dict={_inputs: batch_x,
                                    y: batch_y})
    if i % 1000 == 0:
        acc = sess.run(accuracy, feed_dict={_inputs: batch_x,
                                            y: batch_y})
        loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x,
                                                  y: batch_y})
        print("Iter " + str(i) + ", Minibatch Loss= " +
              "{:.6f}".format(loss) + ", Training Accuracy= " +
              "{:.5f}".format(acc))

print("Testing Accuracy:", sess.run(
    accuracy, feed_dict={_inputs: test_data, y: test_label}))
