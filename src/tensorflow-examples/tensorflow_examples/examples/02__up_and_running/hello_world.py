import tensorflow as tf

h = tf.constant("Hello")
w = tf.constant(" World!")
hw = h + w

with tf.compat.v1.Session() as sess:
    ans = sess.run(hw)

print(ans)
