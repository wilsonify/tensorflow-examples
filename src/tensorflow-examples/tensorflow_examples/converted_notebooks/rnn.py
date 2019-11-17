#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Recurrent Neural Networks (RNN) with Keras
# 
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/keras/rnn">
#     <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
#     View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/keras/rnn.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
#     Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/guide/keras/rnn.ipynb">
#     <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
#     View source on GitHub</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/guide/keras/rnn.ipynb">
#     <img src="https://www.tensorflow.org/images/download_logo_32px.png" />
#     Download notebook</a>
#   </td>
# </table>
# 

# Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language.
# 
# Schematically, a RNN layer uses a `for` loop to iterate over the timesteps of a sequence, while maintaining an internal state that encodes information about the timesteps it has seen so far.
# 
# The Keras RNN API is designed with a focus on:
# 
# - **Ease of use**: the built-in `tf.keras.layers.RNN`, `tf.keras.layers.LSTM`, `tf.keras.layers.GRU` layers enable you to quickly build recurrent models without having to make difficult configuration choices.
#   
# - **Ease of customization**: You can also define your own RNN cell layer (the inner part of the `for` loop) with custom behavior, and use it with the generic `tf.keras.layers.RNN` layer (the `for` loop itself). This allows you to quickly prototype different research ideas in a flexible way with minimal code.
#   

# ## Setup

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras import layers


# ## Build a simple model
# 

# There are three built-in RNN layers in Keras:
# 
# 1. `tf.keras.layers.SimpleRNN`, a fully-connected RNN where the output from previous timestep is to be fed to next timestep.
# 
# 2. `tf.keras.layers.GRU`, first proposed in [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078).
# 
# 3. `tf.keras.layers.LSTM`, first proposed in [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf).
# 
# In early 2015, Keras had the first reusable open-source Python implementations of LSTM and GRU.
# 
# Here is a simple example of a `Sequential` model that processes sequences of integers, embeds each integer into a 64-dimensional vector, then processes the sequence of vectors using a `LSTM` layer.

# In[3]:


model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units and softmax activation.
model.add(layers.Dense(10, activation='softmax'))

model.summary()


# ## Outputs and states

# By default, the output of a RNN layer contain a single vector per sample. This vector is the RNN cell output corresponding to the last timestep, containing information about the entire input sequence. The shape of this output is `(batch_size, units)` where `units` corresponds to the `units` argument passed to the layer's constructor. 
# 
# A RNN layer can also return the entire sequence of outputs for each sample (one vector per timestep per sample), if you set `return_sequences=True`. The shape of this output is `(batch_size, timesteps, units)`.

# In[4]:


model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(10, activation='softmax'))

model.summary() 


# In addition, a RNN layer can return its final internal state(s). The returned states can be used to resume the RNN execution later, or [to initialize another RNN](https://arxiv.org/abs/1409.3215). This setting is commonly used in the encoder-decoder sequence-to-sequence model, where the encoder final state is used as the initial state of the decoder.
# 
# To configure a RNN layer to return its internal state, set the `return_state` parameter to `True` when creating the layer. Note that `LSTM` has 2 state  tensors, but `GRU` only has one.
# 
# To configure the initial state of the layer, just call the layer with additional keyword argument `initial_state`.
# Note that the shape of the state needs to match the unit size of the layer, like in the example below.

# In[5]:


encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None, ))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

# Return states in addition to output
output, state_h, state_c = layers.LSTM(
    64, return_state=True, name='encoder')(encoder_embedded)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None, ))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

# Pass the 2 states to a new LSTM layer, as initial state
decoder_output = layers.LSTM(
    64, name='decoder')(decoder_embedded, initial_state=encoder_state)
output = layers.Dense(10, activation='softmax')(decoder_output)

model = tf.keras.Model([encoder_input, decoder_input], output)
model.summary()


# ## RNN layers and RNN cells

# In addition to the built-in RNN layers, the RNN API also provides cell-level APIs. Unlike RNN layers, which processes whole batches of input sequences, the RNN cell only processes a single timestep.
# 
# The cell is the inside of the `for` loop of a RNN layer. Wrapping a cell inside a `tf.keras.layers.RNN` layer gives you a layer capable of processing batches of sequences, e.g. `RNN(LSTMCell(10))`.
# 
# Mathematically, `RNN(LSTMCell(10))` produces the same result as `LSTM(10)`. In fact, the implementation of this layer in TF v1.x was just creating the corresponding RNN cell and wrapping it in a RNN layer.  However using the built-in `GRU` and `LSTM` layers enables the use of CuDNN and you may see better performance.
# 
# There are three built-in RNN cells, each of them corresponding to the matching RNN layer.
# 
# - `tf.keras.layers.SimpleRNNCell` corresponds to the `SimpleRNN` layer.
# 
# - `tf.keras.layers.GRUCell` corresponds to the `GRU` layer.
# 
# - `tf.keras.layers.LSTMCell` corresponds to the `LSTM` layer.
# 
# The cell abstraction, together with the generic `tf.keras.layers.RNN` class, make it very easy to implement custom RNN architectures for your research.
# 

# ## Cross-batch statefulness

# When processing very long sequences (possibly infinite), you may want to use the pattern of **cross-batch statefulness**.
# 
# Normally, the internal state of a RNN layer is reset every time it sees a new batch (i.e. every sample seen by the layer is assume to be independent from the past). The layer will only maintain a state while processing a given sample.
# 
# If you have very long sequences though, it is useful to break them into shorter sequences, and to feed these shorter sequences sequentially into a RNN layer without resetting the layer's state. That way, the layer can retain information about the entirety of the sequence, even though it's only seeing one sub-sequence at a time.
# 
# You can do this by setting `stateful=True` in the constructor.
# 
# If you have a sequence `s = [t0, t1, ... t1546, t1547]`, you would split it into e.g.
# 
# ```
# s1 = [t0, t1, ... t100]
# s2 = [t101, ... t201]
# ...
# s16 = [t1501, ... t1547]
# ```
# 
# Then you would process it via:
# 
# ```python
# lstm_layer = layers.LSTM(64, stateful=True)
# for s in sub_sequences:
#   output = lstm_layer(s)
# ```
# 
# When you want to clear the state, you  can use `layer.reset_states()`.
# 
# 
# > Note: In this setup, sample `i` in a given batch is assumed to be the continuation of sample `i` in the previous batch. This means that all batches should contain the same number of samples (batch size). E.g. if a batch contains `[sequence_A_from_t0_to_t100,  sequence_B_from_t0_to_t100]`, the next batch should contain `[sequence_A_from_t101_to_t200,  sequence_B_from_t101_to_t200]`.
# 
# 
# 
# 
# Here is a complete example:
# 
# 

# In[6]:


paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)

# reset_states() will reset the cached state to the original initial_state.
# If no initial_state was provided, zero-states will be used by default.
lstm_layer.reset_states()


# ##Bidirectional RNNs

# For sequences other than time series (e.g. text), it is often the case that a RNN model can perform better if it not only processes sequence from start to end, but also backwards. For example, to predict the next word in a sentence, it is often useful to have the context around the word, not only just the words that come before it.
# 
# Keras provides an easy API for you to build such bidirectional RNNs: the `tf.keras.layers.Bidirectional` wrapper.

# In[7]:


model = tf.keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), 
                               input_shape=(5, 10)))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10, activation='softmax'))

model.summary()


# Under the hood, `Bidirectional` will copy the RNN layer passed in, and flip the `go_backwards` field of the newly copied layer, so that it will process the inputs in reverse order.
# 
# The output of the `Bidirectional` RNN will be, by default, the sum of the forward layer output and the backward layer output. If you need a different merging behavior, e.g. concatenation, change the `merge_mode` parameter in the `Bidirectional` wrapper constructor. For more details about `Bidirectional`, please check [the API docs](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Bidirectional).

# ## Performance optimization and CuDNN kernels in TensorFlow 2.0

# In TensorFlow 2.0, the built-in LSTM and GRU layers have been updated to leverage CuDNN kernels by default when a GPU is available. With this change, the prior `keras.layers.CuDNNLSTM/CuDNNGRU` layers have been deprecated, and you can build your model without worrying about the hardware it will run on.
# 
# Since the CuDNN kernel is built with certain assumptions, this means the layer **will not be able to use the CuDNN kernel if you change the defaults of the built-in LSTM or GRU layers**. E.g.:
# 
# - Changing the `activation` function from `tanh` to something else.
# - Changing the `recurrent_activation` function from `sigmoid` to something else.
# - Using `recurrent_dropout` > 0.
# - Setting `unroll` to True, which forces LSTM/GRU to decompose the inner `tf.while_loop` into an unrolled `for` loop.
# - Setting `use_bias` to False.
# - Using masking when the input data is not strictly right padded (if the mask corresponds to strictly right padded data, CuDNN can still be used. This is the most common case).
# 
# For the detailed list of constraints, please see the documentation for the [LSTM](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/LSTM) and [GRU](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/GRU) layers.

# ### Using CuDNN kernels when available
# 
# Let's build a simple LSTM model to demonstrate the performance difference.
# 
# We'll use as input sequences the sequence of rows of MNIST digits (treating each row of pixels as a timestep), and we'll predict the digit's label.
# 

# In[8]:


batch_size = 64
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 28

units = 64
output_size = 10  # labels are from 0 to 9

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
  # CuDNN is only available at the layer level, and not at the cell level.
  # This means `LSTM(units)` will use the CuDNN kernel,
  # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
  if allow_cudnn_kernel:
    # The LSTM layer with default options uses CuDNN.
    lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
  else:
    # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
    lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(units),
        input_shape=(None, input_dim))
  model = tf.keras.models.Sequential([
      lstm_layer,
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(output_size, activation='softmax')]
  )
  return model


# ### Load MNIST dataset

# In[9]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]


# ### Create a model instance and compile it
# We choose `sparse_categorical_crossentropy` as the loss function for the model. The output of the model has shape of `[batch_size, 10]`. The target for the model is a integer vector, each of the integer is in the range of 0 to 9.

# In[10]:


model = build_model(allow_cudnn_kernel=True)

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='sgd',
              metrics=['accuracy'])


# In[11]:


model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=5)


# ### Build a new model without CuDNN kernel

# In[12]:


slow_model = build_model(allow_cudnn_kernel=False)
slow_model.set_weights(model.get_weights())
slow_model.compile(loss='sparse_categorical_crossentropy', 
                   optimizer='sgd', 
                   metrics=['accuracy'])
slow_model.fit(x_train, y_train, 
               validation_data=(x_test, y_test), 
               batch_size=batch_size,
               epochs=1)  # We only train for one epoch because it's slower.


# As you can see, the model built with CuDNN is much faster to train compared to the model that use the regular TensorFlow kernel.
# 
# The same CuDNN-enabled model can also be use to run inference in a CPU-only environment. The `tf.device` annotation below is just forcing the device placement. The model will run on CPU by default if no GPU is available.
# 
# You simply don't have to worry about the hardware you're running on anymore. Isn't that pretty cool?

# In[13]:


with tf.device('CPU:0'):
  cpu_model = build_model(allow_cudnn_kernel=True)
  cpu_model.set_weights(model.get_weights())
  result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
  print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
  plt.imshow(sample, cmap=plt.get_cmap('gray'))


# ## RNNs with list/dict inputs, or nested inputs
# 
# Nested structures allow implementers to include more information within a single timestep. For example, a video frame could have audio and video input at the same time. The data shape in this case could be:
# 
# `[batch, timestep, {"video": [height, width, channel], "audio": [frequency]}]`
# 
# In another example, handwriting data could have both coordinates x and y for the current position of the pen, as well as pressure information. So the data representation could be:
# 
# `[batch, timestep, {"location": [x, y], "pressure": [force]}]`
# 
# The following code provides an example of how to build a custom RNN cell that accepts such structured inputs.
# 

# ### Define a custom cell that support nested input/output

# In[14]:


NestedInput = collections.namedtuple('NestedInput', ['feature1', 'feature2'])
NestedState = collections.namedtuple('NestedState', ['state1', 'state2'])

class NestedCell(tf.keras.layers.Layer):

  def __init__(self, unit_1, unit_2, unit_3, **kwargs):
    self.unit_1 = unit_1
    self.unit_2 = unit_2
    self.unit_3 = unit_3
    self.state_size = NestedState(state1=unit_1, 
                                  state2=tf.TensorShape([unit_2, unit_3]))
    self.output_size = (unit_1, tf.TensorShape([unit_2, unit_3]))
    super(NestedCell, self).__init__(**kwargs)

  def build(self, input_shapes):
    # expect input_shape to contain 2 items, [(batch, i1), (batch, i2, i3)]
    input_1 = input_shapes.feature1[1]
    input_2, input_3 = input_shapes.feature2[1:]

    self.kernel_1 = self.add_weight(
        shape=(input_1, self.unit_1), initializer='uniform', name='kernel_1')
    self.kernel_2_3 = self.add_weight(
        shape=(input_2, input_3, self.unit_2, self.unit_3),
        initializer='uniform',
        name='kernel_2_3')

  def call(self, inputs, states):
    # inputs should be in [(batch, input_1), (batch, input_2, input_3)]
    # state should be in shape [(batch, unit_1), (batch, unit_2, unit_3)]
    input_1, input_2 = tf.nest.flatten(inputs)
    s1, s2 = states

    output_1 = tf.matmul(input_1, self.kernel_1)
    output_2_3 = tf.einsum('bij,ijkl->bkl', input_2, self.kernel_2_3)
    state_1 = s1 + output_1
    state_2_3 = s2 + output_2_3

    output = [output_1, output_2_3]
    new_states = NestedState(state1=state_1, state2=state_2_3)

    return output, new_states


# ### Build a RNN model with nested input/output
# 
# Let's build a Keras model that uses a `tf.keras.layers.RNN` layer and the custom cell we just defined.

# In[15]:


unit_1 = 10
unit_2 = 20
unit_3 = 30

input_1 = 32
input_2 = 64
input_3 = 32
batch_size = 64
num_batch = 100
timestep = 50

cell = NestedCell(unit_1, unit_2, unit_3)
rnn = tf.keras.layers.RNN(cell)

inp_1 = tf.keras.Input((None, input_1))
inp_2 = tf.keras.Input((None, input_2, input_3))

outputs = rnn(NestedInput(feature1=inp_1, feature2=inp_2))

model = tf.keras.models.Model([inp_1, inp_2], outputs)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# ### Train the model with randomly generated data
# 
# Since there isn't a good candidate dataset for this model, we use random Numpy data for demonstration.

# In[16]:


input_1_data = np.random.random((batch_size * num_batch, timestep, input_1))
input_2_data = np.random.random((batch_size * num_batch, timestep, input_2, input_3))
target_1_data = np.random.random((batch_size * num_batch, unit_1))
target_2_data = np.random.random((batch_size * num_batch, unit_2, unit_3))
input_data = [input_1_data, input_2_data]
target_data = [target_1_data, target_2_data]

model.fit(input_data, target_data, batch_size=batch_size)


# With the Keras `tf.keras.layers.RNN` layer, You are only expected to define the math logic for individual step within the sequence, and the `tf.keras.layers.RNN` layer will handle the sequence iteration for you. It's an incredibly powerful way to quickly prototype new kinds of RNNs (e.g. a LSTM variant).
# 
# For more details, please visit the [API docs](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/RNN).
