import tensorflow as tf


def main_1():
    with tf.compat.v1.Session() as sess:
        h = tf.constant("Hello")
        w = tf.constant(" World!")
        hw = h + w
        ans = sess.run(hw)
    return ans


def main_2():
    msg = tf.constant('TensorFlow 2.0 Hello World')
    return tf.print(msg)


if __name__ == "__main__":
    print(main_1())
    print(main_2())
