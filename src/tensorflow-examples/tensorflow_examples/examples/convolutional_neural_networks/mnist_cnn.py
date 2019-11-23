"""
follow mnist example
similar to softmax example, but with convolution added
"""

import logging

import numpy as np
import tensorflow as tf

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
    data_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    data_dir = "/tmp/data"
    path = tf.keras.utils.get_file(
        fname="mnist.npz", origin=data_url, cache_dir=data_dir
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

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset


def construct_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                     activation='relu',
                                     input_shape=INPUT_SHAPE))
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
    train_dataset, test_dataset = load_data()

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
