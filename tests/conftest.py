import pytest
from tensorflow_examples.examples.convolutional_neural_networks import cifar_cnn


@pytest.fixture
def cifar_data_manager():
    return cifar_cnn.CifarDataManager()


@pytest.fixture
def cifar_loader():
    return cifar_cnn.CifarLoader()
