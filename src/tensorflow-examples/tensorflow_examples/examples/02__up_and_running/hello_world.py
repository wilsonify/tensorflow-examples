import tensorflow as tf


def main():
    with tf.compat.v1.Session() as sess:
        h = tf.constant("Hello")
        w = tf.constant(" World!")
        hw = h + w
        ans = sess.run(hw)
    return ans


if __name__ == "__main__":
    ans = main()
    print(ans)
