import numpy as np
import tensorflow as tf
from keras.datasets.mnist import load_data
from tensorflow_examples.layers import conv_layer, max_pool_2x2, full_layer

DATA_DIR = '/tmp/data'
MINIBATCH_SIZE = 50
STEPS = 5000


def main():
    mnist = load_data(path=DATA_DIR)

    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

    keep_prob = tf.compat.v1.placeholder(tf.float32)
    full1_drop = tf.nn.dropout(full_1, rate=1 - (keep_prob))

    y_conv = full_layer(full1_drop, 10)

    cross_entropy = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=tf.stop_gradient(y_)))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(input=y_conv, axis=1), tf.argmax(input=y_, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(STEPS):
            batch = mnist.train.next_batch(MINIBATCH_SIZE)

            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1],
                                                               keep_prob: 1.0})
                print("step {}, training accuracy {}".format(i, train_accuracy))

            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        X = mnist.test.images.reshape(10, 1000, 784)
        Y = mnist.test.labels.reshape(10, 1000, 10)
        test_accuracy = np.mean(
            [sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])

    print("test accuracy: {}".format(test_accuracy))


if __name__ == "__main__":
    main()
