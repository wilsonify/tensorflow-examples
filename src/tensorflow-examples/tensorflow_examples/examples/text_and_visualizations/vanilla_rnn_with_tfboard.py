import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

num_classes = 10  # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).
learning_rate = 0.001
training_steps = 1000
batch_size = 32
display_step = 100
num_input = 28
timesteps = 28
num_units = 32


class LSTM(Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_layer = layers.LSTM(units=num_units)
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = self.lstm_layer(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x


def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def run_optimization(x, y, lstm_net, optimizer):
    with tf.GradientTape() as g:
        pred = lstm_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)
    trainable_variables = lstm_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    x_train, x_test = x_train.reshape([-1, 28, 28]), x_test.reshape(
        [-1, num_features])  # Flatten images to 1-D vector of 784 features (28*28).
    x_train, x_test = x_train / 255., x_test / 255.  # Normalize images value from [0, 255] to [0, 1].

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

    lstm_net = LSTM()
    optimizer = tf.optimizers.Adam(learning_rate)

    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        run_optimization(batch_x, batch_y, lstm_net=lstm_net, optimizer=optimizer)
        if step % display_step == 0:
            pred = lstm_net(batch_x, is_training=True)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)
            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))


if __name__ == "__main__":
    main()
