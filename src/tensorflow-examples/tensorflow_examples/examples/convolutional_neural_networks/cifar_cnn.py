import os
import pickle
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_examples.layers import conv_layer, max_pool_2x2, full_layer

HOME_DIR = os.path.expanduser("~")
ARCHIVE_PATH = os.path.join(HOME_DIR, "Downloads",
                            "cifar-10-python.tar.gz")  # manually download and extract https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
DATA_DIR = os.path.join(HOME_DIR, "Downloads", "cifar-10-python")
BATCH_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
BATCH_SIZE = 50
STEPS = 500000


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def unzip(file_name, destination_dir=DATA_DIR):
    if file_name.endswith(".tar.gz"):
        extractor = tarfile.open
        compression_type = 'gz'
        read_mode = "r:{}".format(compression_type)
    elif file_name.endswith(".tar"):
        extractor = tarfile.open
        compression_type = ''
        read_mode = "r:{}".format(compression_type)
    elif file_name.endswith(".zip"):
        import zipfile
        extractor = zipfile.ZipFile
        read_mode = 'r'

    open_args = [file_name, read_mode]
    with extractor(*open_args) as ref:
        ref.extractall(path=destination_dir)


def unpickle(file):
    with open(os.path.join(BATCH_DIR, file), 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict


def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()


class CifarLoader(object):
    """
    Load and mange the CIFAR dataset.
    (for any practical use there is no reason not to use the built-in dataset handler instead)
    """

    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1) \
                          .astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i + batch_size], \
               self.labels[self._i:self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]) \
            .load()
        self.test = CifarLoader(["test_batch"]).load()


def run_simple_net():
    cifar = CifarDataManager()
    tf.compat.v1.disable_eager_execution()

    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    conv1 = conv_layer(x, shape=[5, 5, 3, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)

    conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 128])
    conv3_pool = max_pool_2x2(conv3)
    conv3_flat = tf.reshape(conv3_pool, [-1, 4 * 4 * 128])
    conv3_drop = tf.nn.dropout(conv3_flat, rate=1 - (keep_prob))

    full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
    full1_drop = tf.nn.dropout(full_1, rate=1 - (keep_prob))

    y_conv = full_layer(full1_drop, 10)

    cross_entropy = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                                                                                        labels=tf.stop_gradient(y_)))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(input=y_conv, axis=1), tf.argmax(input=y_, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

    def test(sess):
        X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
        Y = cifar.test.labels.reshape(10, 1000, 10)
        acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})
                       for i in range(10)])
        print("Accuracy: {:.4}%".format(acc * 100))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(STEPS):
            batch = cifar.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % 500 == 0:
                test(sess)

        test(sess)


def build_second_net():
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    C1, C2, C3 = 32, 64, 128
    F1 = 600

    conv1_1 = conv_layer(x, shape=[3, 3, 3, C1])
    conv1_2 = conv_layer(conv1_1, shape=[3, 3, C1, C1])
    conv1_3 = conv_layer(conv1_2, shape=[3, 3, C1, C1])
    conv1_pool = max_pool_2x2(conv1_3)
    conv1_drop = tf.nn.dropout(conv1_pool, rate=1 - (keep_prob))

    conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C1, C2])
    conv2_2 = conv_layer(conv2_1, shape=[3, 3, C2, C2])
    conv2_3 = conv_layer(conv2_2, shape=[3, 3, C2, C2])
    conv2_pool = max_pool_2x2(conv2_3)
    conv2_drop = tf.nn.dropout(conv2_pool, rate=1 - (keep_prob))

    conv3_1 = conv_layer(conv2_drop, shape=[3, 3, C2, C3])
    conv3_2 = conv_layer(conv3_1, shape=[3, 3, C3, C3])
    conv3_3 = conv_layer(conv3_2, shape=[3, 3, C3, C3])
    conv3_pool = tf.nn.max_pool2d(input=conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    conv3_flat = tf.reshape(conv3_pool, [-1, C3])
    conv3_drop = tf.nn.dropout(conv3_flat, rate=1 - (keep_prob))

    full1 = tf.nn.relu(full_layer(conv3_drop, F1))
    full1_drop = tf.nn.dropout(full1, rate=1 - (keep_prob))

    y_conv = full_layer(full1_drop, 10)

    cross_entropy = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                                                                                        labels=tf.stop_gradient(y_)))
    train_step = tf.compat.v1.train.AdamOptimizer(5e-4).minimize(cross_entropy)  # noqa

    correct_prediction = tf.equal(tf.argmax(input=y_conv, axis=1), tf.argmax(input=y_, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))  # noqa

    # Plug this into the test procedure as above to continue...


def create_cifar_image():
    d = CifarDataManager()
    print("Number of train images: {}".format(len(d.train.images)))
    print("Number of train labels: {}".format(len(d.train.labels)))
    print("Number of test images: {}".format(len(d.test.images)))
    print("Number of test labels: {}".format(len(d.test.labels)))
    images = d.train.images
    display_cifar(images, 10)


if __name__ == "__main__":
    create_cifar_image()
    run_simple_net()
