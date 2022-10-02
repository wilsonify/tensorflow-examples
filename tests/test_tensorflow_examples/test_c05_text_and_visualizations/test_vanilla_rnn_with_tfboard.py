import datetime
import logging

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

num_classes = 10  # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).
learning_rate = 0.001
training_steps = 1000
batch_size = 32
display_step = 100
num_input = 28
timesteps = 28
num_units = 32

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100
IMAGE_SHAPE = (28, 28)
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
EPOCHS = 5


def load_data():
    """
    get and split data
    :return: tf.data.Dataset suitable for keras flows
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    sample, sample_label = x_train[0], y_train[0]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset


def main():
    train_dataset, test_dataset = load_data()

    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))
    model.add(layers.GRU(256, return_sequences=True))
    model.add(layers.SimpleRNN(128))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        train_dataset,
        epochs=5,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback]
    )


if __name__ == "__main__":
    main()
