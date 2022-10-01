import pytest

from tensorflow_examples import cifar_cnn


@pytest.fixture
def cifar_data_manager():
    return cifar_cnn.CifarDataManager()


@pytest.fixture
def cifar_loader():
    return cifar_cnn.CifarLoader()
