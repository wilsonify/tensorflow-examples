"""
follow mnist example
similar to softmax example, but with convolution added
"""

import logging
import os

import numpy as np
import tensorflow as tf

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100
IMAGE_SHAPE = (28, 28)
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
EPOCHS = 5


class MnistLoader(object):
    """
    Load and mange the MNIST dataset.
    """

    def __init__(self, data_dir="/tmp/data/datasets"):
        self.data_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        self.data_dir = "/tmp/data/datasets"
        self._source = self.data_dir
        self._i = 0
        self.train_dataset = None
        self.test_dataset = None
        self.images = None
        self.labels = None
        self.test_images = None
        self.test_labels = None

    def load_data(self):
        """
        get and split data
        :return: tf.data.Dataset suitable for keras flows
        """
        os.makedirs(self.data_dir, exist_ok=True)
        path = tf.keras.utils.get_file(
            fname="mnist.npz",
            origin=self.data_url,
            cache_dir=self.data_dir
        )

        with np.load(path) as data:
            train_examples = data["x_train"]
            train_examples = train_examples.reshape(-1, 28, 28, 1)
            train_labels = tf.keras.utils.to_categorical(data["y_train"], NUM_CLASSES)
            test_examples = data["x_test"]
            test_labels = tf.keras.utils.to_categorical(data["y_test"], NUM_CLASSES)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
        logging.debug("%r", "type(train_dataset) = {}".format(type(train_dataset)))
        logging.debug("%r", "type(test_dataset) = {}".format(type(test_dataset)))
        self.train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.test_dataset = test_dataset.batch(BATCH_SIZE)
        self.images = train_examples
        self.labels = train_labels
        self.test_images = test_examples
        self.test_labels = test_labels
        return self

    def next_batch(self, batch_size=BATCH_SIZE):
        x, y = self.images[self._i:self._i + batch_size], \
               self.labels[self._i:self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def random_batch(self, batch_size=BATCH_SIZE):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]


def construct_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    return model


def main():
    """
    main function
    load, train, evaluate
    :return:
    """
    # input image dimensions
    mnist_loaded = MnistLoader()
    mnist_loaded.load_data()
    train_dataset = mnist_loaded.train_dataset
    test_dataset = mnist_loaded.test_dataset

    model = construct_model()

    compile_kwargs = {
        'loss': tf.keras.losses.categorical_crossentropy,
        'optimizer': tf.keras.optimizers.Adadelta(),
        'metrics': ['accuracy']
    }

    model.compile(**compile_kwargs)

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        verbose=2,
    )
    score = model.evaluate(test_dataset, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    main()
