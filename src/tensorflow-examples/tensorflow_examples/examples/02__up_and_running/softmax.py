import tensorflow as tf
from tensorflow_examples.tutorials.mnist import input_data


DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100


data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.compat.v1.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.compat.v1.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=tf.stop_gradient(y_true)))

gd_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_true, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_mask, tf.float32))

with tf.compat.v1.Session() as sess:

    # Train
    sess.run(tf.compat.v1.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))
