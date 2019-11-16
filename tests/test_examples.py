import pytest
from tensorflow_examples.examples import (
    convolutional_neural_networks,
    distributed_tensorflow,
    queues_threads,
    text_and_visualizations,
    up_and_running,
    word_embeddings_and_rnns
)
from tensorflow_examples.examples.convolutional_neural_networks import cifar_cnn, mnist_cnn


def test_smoke():
    print("is anything on fire")


@pytest.mark.skip(reason='not implemented yet')
def test_distribute():
    distributed_tensorflow.distribute()


@pytest.mark.skip(reason='not implemented yet')
def test_distribute_run():
    distributed_tensorflow.distribute_run()


@pytest.mark.skip(reason='not implemented yet')
def test_queue_basic():
    queues_threads.queue_basic()


@pytest.mark.skip(reason='not implemented yet')
def test_tfrecords_end_to_end():
    queues_threads.tfrecords_end_to_end()


@pytest.mark.skip(reason='not implemented yet')
def test_tfrecords_read_write():
    queues_threads.tfrecords_read_write()


@pytest.mark.skip(reason='not implemented yet')
def test_BasicRNNCell():
    text_and_visualizations.BasicRNNCell()


@pytest.mark.skip(reason='not implemented yet')
def test_LSTM_supervised_embeddings():
    text_and_visualizations.LSTM_supervised_embeddings()


@pytest.mark.skip(reason='not implemented yet')
def test_scan_example():
    text_and_visualizations.scan_example()


@pytest.mark.skip(reason='not implemented yet')
def test_vanilla_rnn_with_tfboard():
    text_and_visualizations.vanilla_rnn_with_tfboard()


@pytest.mark.skip(reason='not implemented yet')
def test_softmax():
    up_and_running.softmax()


@pytest.mark.skip(reason='not implemented yet')
def test_GRU_pretrained_GloVe():
    word_embeddings_and_rnns.GRU_pretrained_GloVe()


@pytest.mark.skip(reason='not implemented yet')
def test_word2vec():
    word_embeddings_and_rnns.word2vec()


@pytest.mark.skip(reason='not implemented yet')
def test_build_second_net(cifar_data_manager):
    cifar_data_manager.build_second_net()


@pytest.mark.skip(reason='not implemented yet')
def test_create_cifar_image():
    convolutional_neural_networks.cifar_cnn.CifarDataManager.create_cifar_image()


@pytest.mark.skip(reason='not implemented yet')
def test_display_cifar():
    convolutional_neural_networks.cifar_cnn.CifarDataManager.display_cifar()


@pytest.mark.skip(reason='not implemented yet')
def test_one_hot():
    convolutional_neural_networks.cifar_cnn.CifarDataManager.one_hot()


@pytest.mark.skip(reason='not implemented yet')
def test_run_simple_net():
    convolutional_neural_networks.cifar_cnn.CifarDataManager.run_simple_net()


@pytest.mark.skip(reason='not implemented yet')
def test_unpickle():
    convolutional_neural_networks.cifar_cnn.CifarDataManager.unpickle()


@pytest.mark.skip(reason='not implemented yet')
def test_mnist_cnn():
    convolutional_neural_networks.mnist_cnn()


@pytest.mark.skip(reason='not implemented yet')
def test_distribute():
    distributed_tensorflow.distribute()


@pytest.mark.skip(reason='not implemented yet')
def test_distribute_run():
    distributed_tensorflow.distribute_run()


@pytest.mark.skip(reason='not implemented yet')
def test_queue_basic():
    queues_threads.queue_basic()


@pytest.mark.skip(reason='not implemented yet')
def test_tfrecords_end_to_end():
    queues_threads.tfrecords_end_to_end()


@pytest.mark.skip(reason='not implemented yet')
def test_tfrecords_read_write():
    queues_threads.tfrecords_read_write()


@pytest.mark.skip(reason='not implemented yet')
def test_BasicRNNCell():
    text_and_visualizations.BasicRNNCell()


@pytest.mark.skip(reason='not implemented yet')
def test_LSTM_supervised_embeddings():
    text_and_visualizations.LSTM_supervised_embeddings()


@pytest.mark.skip(reason='not implemented yet')
def test_get_sentence_batch():
    text_and_visualizations.LSTM_supervised_embeddings.get_sentence_batch()


@pytest.mark.skip(reason='not implemented yet')
def test_scan_example():
    text_and_visualizations.scan_example()


@pytest.mark.skip(reason='not implemented yet')
def test_vanilla_rnn_with_tfboard():
    text_and_visualizations.vanilla_rnn_with_tfboard()


@pytest.mark.skip(reason='not implemented yet')
def test_hello_world_main_1():
    from tensorflow_examples.examples.up_and_running import hello_world
    up_and_running.hello_world.main_1()


@pytest.mark.skip(reason='not implemented yet')
def test_hello_world_main_2():
    from tensorflow_examples.examples.up_and_running import hello_world
    hello_world.main_2()


@pytest.mark.skip(reason='not implemented yet')
def test_softmax():
    up_and_running.softmax()


@pytest.mark.skip(reason='not implemented yet')
def test_GRU_pretrained_GloVe():
    word_embeddings_and_rnns.GRU_pretrained_GloVe()


@pytest.mark.skip(reason='not implemented yet')
def test_word2vec():
    word_embeddings_and_rnns.word2vec()
