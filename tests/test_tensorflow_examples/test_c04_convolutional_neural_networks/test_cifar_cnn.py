from tensorflow_examples.c04_convolutional_neural_networks.cifar_cnn import (
    create_cifar_image,
    run_simple_net
)


def test_create_cifar_image():
    print("create cifar image")
    create_cifar_image()
    print("done with create cifar image")


def test_run_simple_net():
    # create_cifar_image()
    print("run simple net")
    run_simple_net()
    print("done with run simple net")
