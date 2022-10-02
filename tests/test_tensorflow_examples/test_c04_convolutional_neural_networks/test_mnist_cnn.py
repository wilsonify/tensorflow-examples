"""
follow mnist example
similar to softmax example, but with convolution added
"""

import tensorflow as tf

from tensorflow_examples.c04_convolutional_neural_networks.mnist_cnn import load_data, construct_model

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100
IMAGE_SHAPE = (28, 28)
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
EPOCHS = 5
compile_kwargs = {
    'loss': tf.keras.losses.categorical_crossentropy,
    'optimizer': tf.keras.optimizers.Adadelta(),
    'metrics': ['accuracy']
}


def test_load_data():
    train_dataset, test_dataset = load_data()


def test_construct_model():
    train_dataset, test_dataset = load_data()
    model = construct_model()


def test_model_compile():
    train_dataset, test_dataset = load_data()
    model = construct_model()
    model.compile(**compile_kwargs)


def test_model_fit():
    train_dataset, test_dataset = load_data()
    model = construct_model()
    model.compile(**compile_kwargs)
    model.fit(train_dataset, epochs=EPOCHS, verbose=2, )


def test_model_evaluate():
    train_dataset, test_dataset = load_data()
    model = construct_model()
    model.compile(**compile_kwargs)
    model.fit(train_dataset, epochs=EPOCHS, verbose=2, )
    score = model.evaluate(test_dataset, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
