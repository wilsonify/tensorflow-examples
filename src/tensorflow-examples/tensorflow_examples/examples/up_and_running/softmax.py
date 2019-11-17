import datetime
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
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=x_train,
              y=y_train,
              epochs=5,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])


if __name__ == "__main__":
    dictConfig(config.LOGGING_CONFIG_DICT)
    main()
