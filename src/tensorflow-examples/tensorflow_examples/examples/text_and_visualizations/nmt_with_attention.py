#!/usr/bin/env python
# coding: utf-8
"""
Neural machine translation with attention

trains a sequence to sequence (seq2seq) model for Spanish to English translation.

input a Spanish sentence, such as *"¿todavia estan en casa?"*,
and return the English translation: *"are you still at home?"*

The translation quality is reasonable.
The attention plot, showing which parts of the input has the model's attention, is more interesting.

<img src="https://tensorflow.org/images/spanish-english.png" alt="spanish-english attention plot">
Note: This example takes approximately 10 mintues to run on a single P100 GPU.
"""

import io
import logging
import os
import re
import time
import unicodedata
from logging.config import dictConfig

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_examples import config
from tensorflow_examples.examples.convolutional_neural_networks.cifar_cnn import unzip

BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024
NUM_EXAMPLES = 30000
NUM_ATTENTION_UNITS = 10
EPOCHS = 10


def max_length(tensor):
    logging.debug("max_length")
    return max(len(t) for t in tensor)


def tokenize(lang):
    logging.debug("tokenize")
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    logging.info("load_dataset")
    logging.info("creating cleaned input, output pairs")
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def create_dataset(path, num_examples):
    """
    1. Remove the accents
    2. Clean the sentences
    3. Return word pairs in the format: [ENGLISH, SPANISH]
    """
    logging.info("create_dataset")
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)


def unicode_to_ascii(s):
    """
    Converts the unicode file to ascii
    :param s:
    :return:
    """
    logging.info("unicode_to_ascii")
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    logging.info("preprocess_sentence")
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r"""[" ]+""", " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def convert(lang, tensor):
    logging.info("convert")
    logging.debug("%r", "lang = {}".format(lang))
    logging.debug("%r", "tensor = {}".format(tensor))
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        logging.info("initialize Encoder")
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        logging.debug("call")
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        logging.debug("initialize_hidden_state")
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        logging.info("initialize BahdanauAttention")
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        logging.debug("call")
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        logging.info("Decoder")
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        logging.debug("call")
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention.call(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred, loss_object):
    logging.info("loss_function")
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, targ_lang, encoder, enc_hidden, optimizer, decoder, loss_object):
    logging.debug("train_step")
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions, loss_object=loss_object)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(sentence, max_length_targ, max_length_inp, inp_lang, encoder, targ_lang, decoder):
    logging.info("evaluate")
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input,
            dec_hidden,
            enc_out
        )

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    """
    function for plotting the attention weights

    :param attention:
    :param sentence:
    :param predicted_sentence:
    :return:
    """
    logging.info("plot_attention")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence, max_length_targ, max_length_inp, inp_lang, encoder, targ_lang, decoder):
    logging.info("translate")
    result, sentence, attention_plot = evaluate(
        sentence,
        max_length_targ=max_length_targ,
        max_length_inp=max_length_inp,
        inp_lang=inp_lang,
        encoder=encoder,
        targ_lang=targ_lang,
        decoder=decoder
    )

    print('Input: %s' % sentence)
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


def main():
    """
    Download dataset
    prepare the dataset
    We'll use a language dataset provided by http://www.manythings.org/anki/.
    This dataset contains language translation pairs in the format:
    ```
    May I borrow this book?	¿Puedo tomar prestado este libro?
    ```
    There are a variety of languages available, but we'll use the English-Spanish dataset.
    For convenience, we've hosted a copy of this dataset on Google Cloud,
    but you can also download your own copy.
    After downloading the dataset, here are the steps we'll take to prepare the data:

    1. Add a *start* and *end* token to each sentence.
    2. Clean the sentences by removing special characters.
    3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
    4. Pad each sentence to a maximum length.
    """
    logging.info("main")
    logging.info("download dataset")
    path_to_zip = tf.keras.utils.get_file(
        fname=os.path.join(config.DATA_DIR, 'spa-eng.zip'),
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True
    )
    logging.info("done downloading dataset")

    logging.info("extract dataset")
    unzip(path_to_zip, destination_dir=config.DATA_DIR)
    logging.info("done extracting dataset")

    path_to_file = os.path.join(os.path.dirname(path_to_zip), "spa-eng", "spa.txt")

    logging.info("test sample sentence")
    en_sentence = u"May I borrow this book?"
    sp_sentence = u"¿Puedo tomar prestado este libro?"
    logging.debug("%r", "en_sentence = {}".format(en_sentence))
    logging.debug("%r", "sp_sentence = {}".format(sp_sentence))
    logging.debug("%r", "preprocess_sentence(en_sentence) = {}".format(preprocess_sentence(en_sentence)))
    logging.debug("%r", "preprocess_sentence(sp_sentence) = {}".format(preprocess_sentence(sp_sentence)))
    logging.info("done with sample sentence")

    logging.info("construct larger dataset")
    en, sp = create_dataset(path_to_file, None)
    logging.debug("en[-1] = {}".format(en[-1]))
    logging.debug("sp[-1] = {}".format(sp[-1]))
    logging.info("done constructing larger dataset")

    logging.info("Limit the size of the dataset to experiment faster (optional)")
    logging.info(
        "Training on the complete dataset of >100,000 sentences will take a long time.\
         To train faster, we can limit the size of the dataset to 30,000 sentences \
         (of course, translation quality degrades with less data):"
    )

    logging.info("Try experimenting with the size of that dataset")

    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, NUM_EXAMPLES)

    logging.info("Calculate max_length of the target tensors")
    max_length_targ = max_length(target_tensor)
    max_length_inp = max_length(input_tensor)

    logging.info("Creating training and validation sets using an 80-20 split")
    (input_tensor_train,
     input_tensor_val,
     target_tensor_train,
     target_tensor_val) = train_test_split(
        input_tensor,
        target_tensor,
        test_size=0.2
    )

    logging.info("Show length")
    logging.debug("%r", "len(input_tensor_train) = {}".format(len(input_tensor_train)))
    logging.debug("%r", "len(target_tensor_train) = {}".format(len(target_tensor_train)))
    logging.debug("%r", "len(input_tensor_val) = {}".format(len(input_tensor_val)))
    logging.debug("%r", "len(target_tensor_val) = {}".format(len(target_tensor_val)))

    print("Input Language; index to word mapping")
    convert(inp_lang, input_tensor_train[0])
    print()
    print("Target Language; index to word mapping")
    convert(targ_lang, target_tensor_train[0])

    logging.info("Create a tf.data dataset")
    buffer_size = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)
    ).shuffle(buffer_size).batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))
    logging.debug("%r", "example_input_batch.shape = {}".format(example_input_batch.shape))
    logging.debug("%r", "example_target_batch.shape = {}".format(example_target_batch.shape))

    logging.info("Write the encoder and decoder model")

    logging.info(
        "Implement an encoder-decoder model with attention which you can read about in the TensorFlow \
        [Neural Machine Translation (seq2seq) tutorial](https://github.com/tensorflow/nmt). \
        This example uses a more recent set of APIs. \
        This notebook implements the \
        [attention equations](https://github.com/tensorflow/nmt#background-on-the-attention-mechanism)\
        from the seq2seq tutorial.\
        The following diagram shows that each input words is assigned a weight by the attention mechanism \
        which is then used by the decoder to predict the next word in the sentence.\
        The below picture and formulas are an example of attention mechanism from \
        [Luong's paper](https://arxiv.org/abs/1508.04025v5).")

    logging.info(
        "The input is put through an encoder model which gives us the encoder output of shape \
        *(batch_size, max_length, hidden_size)*\
         and the encoder hidden state of shape \
         *(batch_size, hidden_size)*.")

    # pseudo-code:
    #
    # * `score = FC(tanh(FC(EO) + FC(H)))`
    # * `attention weights = softmax(score, axis = 1)`.
    #   Softmax by default is applied on the last axis but here we want to apply it on the *1st axis*,
    #   since the shape of score is *(batch_size, max_length, hidden_size)*.
    #   `Max_length` is the length of our input.
    #   Since we are trying to assign a weight to each input, softmax should be applied on that axis.
    # * `context vector = sum(attention weights * EO, axis = 1)`. Same reason as above for choosing axis as 1.
    # * `embedding output` = The input to the decoder X is passed through an embedding layer.
    # * `merged vector = concat(embedding output, context vector)`
    # * This merged vector is then given to the GRU
    #
    # The shapes of all the vectors at each step have been specified in the comments in the code:

    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    logging.info("sample input")
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder.call(example_input_batch, sample_hidden)
    logging.debug("%r", "Encoder output shape: (batch size, sequence length, units) {}".format(sample_output.shape))
    logging.debug("%r", "Encoder Hidden state shape: (batch size, units) {}".format(sample_hidden.shape))

    attention_layer = BahdanauAttention(units=NUM_ATTENTION_UNITS)
    attention_result, attention_weights = attention_layer.call(sample_hidden, sample_output)

    logging.debug("%r", "attention_result.shape = {} should be (batch size, units)".format(attention_result.shape))
    logging.debug("%r", "attention_weights.shape = {} should be (batch_size, sequence_length, 1)".format(
        attention_weights.shape))

    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    sample_decoder_output, _, _ = decoder.call(
        tf.random.uniform((64, 1)),
        sample_hidden,
        sample_output
    )

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

    logging.info("Define the optimizer and the loss function")

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    logging.info("Checkpoints (Object-based saving)")

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    logging.info("Training")
    #
    # 1. Pass the *input* through the *encoder* which return *encoder output* and the *encoder hidden state*.
    # 2. The encoder output, encoder hidden state and the decoder input
    #       (which is the *start token*) is passed to the decoder.
    # 3. The decoder returns the *predictions* and the *decoder hidden state*.
    # 4. The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
    # 5. Use *teacher forcing* to decide the next input to the decoder.
    # 6. *Teacher forcing* is the technique where the *target word* is passed as the *next input* to the decoder.
    # 7. The final step is to calculate the gradients and apply it to the optimizer and backpropagate.

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(
                inp,
                targ=targ,
                targ_lang=targ_lang,
                encoder=encoder,
                enc_hidden=enc_hidden,
                optimizer=optimizer,
                decoder=decoder,
                loss_object=loss_object
            )
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    logging.info("Translate")
    #
    # The evaluate function is similar to the training loop, except we don't use *teacher forcing* here.
    # The input to the decoder at each time step is its previous predictions
    # along with the hidden state and the encoder output.
    # * Stop predicting when the model predicts the *end token*.
    # * And store the *attention weights for every time step*.
    #
    # Note: The encoder output is calculated only once for one input.

    # ## Restore the latest checkpoint and test

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    translate(
        u'hace mucho frio aqui.',
        max_length_targ=max_length_targ,
        max_length_inp=max_length_inp,
        inp_lang=inp_lang,
        encoder=encoder,
        targ_lang=targ_lang,
        decoder=decoder
    )

    translate(
        u'esta es mi vida.',
        max_length_targ=max_length_targ,
        max_length_inp=max_length_inp,
        inp_lang=inp_lang,
        encoder=encoder,
        targ_lang=targ_lang,
        decoder=decoder
    )

    translate(
        u'¿todavia estan en casa?',
        max_length_targ=max_length_targ,
        max_length_inp=max_length_inp,
        inp_lang=inp_lang,
        encoder=encoder,
        targ_lang=targ_lang,
        decoder=decoder

    )

    # wrong translation
    translate(
        u'trata de averiguarlo.',
        max_length_targ=max_length_targ,
        max_length_inp=max_length_inp,
        inp_lang=inp_lang,
        encoder=encoder,
        targ_lang=targ_lang,
        decoder=decoder
    )


if __name__ == "__main__":
    dictConfig(config.LOGGING_CONFIG_DICT)
    main()
