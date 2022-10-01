from pprint import pprint

import tensorflow as tf
from tensorflow_examples import (
    GRU_pretrained_GloVe,
    word2vec,
    mnist_cnn,
    scan_example,
    config,
    layers,
    cifar_cnn,
    hello_world,
    tfrecords_read_write,
    tfrecords_end_to_end,
    distribute_run,
    queue_basic,
    distribute,
    BasicRNNCell,
    LSTM_supervised_embeddings,
    softmax,
    vanilla_rnn_with_tfboard,

)


def test_smoke():
    print("is anything on fire")


def test_config():
    pprint(f"config.LOGGING_CONFIG_DICT = {config.LOGGING_CONFIG_DICT}")
    pprint(f"config.DATA_DIR = {config.DATA_DIR}")
    pprint(f"config.HOME_DIR = {config.HOME_DIR}")


def test_layers():
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 3])
    c1 = 32
    conv1_1 = layers.conv_layer(x, shape=[3, 3, 3, c1])
    assert str(type(conv1_1)) == int


def test_cifar_cnn():
    cifar_cnn.main()


def test_word2vec():
    word2vec.main()


def test_queue_basic():
    queue_basic.main()


def test_tfrecords_read_write():
    tfrecords_read_write.main()


def test_lstm_supervised_embeddings():
    LSTM_supervised_embeddings.main()


def test_scan_example():
    scan_example.main()


def test_vanilla_rnn_with_tfboard():
    vanilla_rnn_with_tfboard.main()


def test_softmax():
    softmax.main()


def test_gru_pretrained_glove():
    GRU_pretrained_GloVe.main()


def test_build_second_net(cifar_data_manager):
    cifar_data_manager.build_second_net()


def test_create_cifar_image():
    cifar_cnn.create_cifar_image()


def test_display_cifar(cifar_data_manager):
    d = cifar_data_manager
    print("Number of train images: {}".format(len(d.train.images)))
    print("Number of train labels: {}".format(len(d.train.labels)))
    print("Number of test images: {}".format(len(d.test.images)))
    print("Number of test labels: {}".format(len(d.test.labels)))
    images = d.train.images
    cifar_cnn.display_cifar(images, 10)


def test_one_hot():
    cdm = cifar_cnn.CifarDataManager()
    cdm.train()
    cdm.test()


def test_run_simple_net():
    cifar_cnn.run_simple_net()


def test_unpickle():
    cifar_cnn.unpickle(file="")


def test_mnist_cnn():
    mnist_cnn.main()


def test_distribute_main():
    distribute.main()


def test_distribute_run_main():
    distribute_run.main()


def test_tfrecords_end_to_end():
    tfrecords_end_to_end.main()


def test_basic_rnn_cell():
    BasicRNNCell.main()


def test_hello_world_main_1():
    hello_world.main_1()


def test_hello_world_main_2():
    hello_world.main_2()
