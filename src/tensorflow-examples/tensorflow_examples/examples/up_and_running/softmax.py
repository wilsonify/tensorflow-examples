import logging
from logging.config import dictConfig

import numpy as np
import tensorflow as tf
from tensorflow_examples import config

NUM_STEPS = 10
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
IMAGE_SHAPE = (28, 28)


def load_data():
    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    DATA_DIR = '/tmp/data'
    path = tf.keras.utils.get_file(
        fname='mnist.npz',
        origin=DATA_URL,
        cache_dir=DATA_DIR
    )
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    logging.debug("%r", "type(train_dataset) = {}".format(type(train_dataset)))
    logging.debug("%r", "type(test_dataset) = {}".format(type(test_dataset)))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, test_dataset


def main():
    train_dataset, test_dataset = load_data()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=IMAGE_SHAPE),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model.fit(train_dataset, epochs=NUM_STEPS)

    model.evaluate(test_dataset)


if __name__ == "__main__":
    dictConfig(config.LOGGING_CONFIG_DICT)
    main()
