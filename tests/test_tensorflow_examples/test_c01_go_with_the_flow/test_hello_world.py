import tensorflow as tf


def test_main_1():
    tf.compat.v1.disable_eager_execution()
    h = tf.constant("Hello")
    w = tf.constant(" World!")
    hw = h + w
    with tf.compat.v1.Session() as sess:
        ans = sess.run(hw)
    return ans


def test_main_2():
    msg = tf.constant('TensorFlow 2.0 Hello World')
    return tf.print(msg)
