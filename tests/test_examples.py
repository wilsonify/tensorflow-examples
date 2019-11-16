from tensorflow_examples.examples import (
    convolutional_neural_networks,
    distributed_tensorflow,
    queues_threads,
    text_and_visualizations,
    up_and_running,
    word_embeddings_and_rnns
)


def test_smoke():
    print("is anything on fire")


def test_cifar_cnn():
    convolutional_neural_networks.cifar_cnn()


def test_mnist_cnn():
    convolutional_neural_networks.mnist_cnn()


def test_distribute():
    distributed_tensorflow.distribute()


def test_distribute_run():
    distributed_tensorflow.distribute_run()


def test_queue_basic():
    queues_threads.queue_basic()


def test_tfrecords_end_to_end():
    queues_threads.tfrecords_end_to_end()


def test_tfrecords_read_write():
    queues_threads.tfrecords_read_write()


def test_BasicRNNCell():
    text_and_visualizations.BasicRNNCell()


def test_LSTM_supervised_embeddings():
    text_and_visualizations.LSTM_supervised_embeddings()


def test_scan_example():
    text_and_visualizations.scan_example()


def test_vanilla_rnn_with_tfboard():
    text_and_visualizations.vanilla_rnn_with_tfboard()


def test_hello_world():
    up_and_running.hello_world()


def test_softmax():
    up_and_running.softmax()


def test_GRU_pretrained_GloVe():
    word_embeddings_and_rnns.GRU_pretrained_GloVe()


def test_word2vec():
    word_embeddings_and_rnns.word2vec()
