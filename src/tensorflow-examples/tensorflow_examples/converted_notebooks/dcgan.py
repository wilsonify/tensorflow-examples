#!/usr/bin/env python
# coding: utf-8
"""

# # Deep Convolutional Generative Adversarial Network

# This tutorial demonstrates how to generate images of handwritten digits using a
[Deep Convolutional Generative Adversarial Network](https://arxiv.org/pdf/1511.06434.pdf) (DCGAN).
The code is written using the
[Keras Sequential API](https://www.tensorflow.org/guide/keras) with a `tf.GradientTape` training loop.

# ## What are GANs?
# [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (GANs)
are one of the most interesting ideas in computer science today.
 Two models are trained simultaneously by an adversarial process.
 A *generator* ("the artist") learns to create images that look real,
 while a *discriminator* ("the art critic") learns to tell real images apart from fakes.
# ![A diagram of a generator and discriminator](./images/gan1.png)
# During training, the *generator* progressively becomes better at creating images that look real,
while the *discriminator* becomes better at telling them apart.
 The process reaches equilibrium when the *discriminator* can no longer distinguish real images from fakes.
# ![A second diagram of a generator and discriminator](./images/gan2.png)
# This notebook demonstrates this process on the MNIST dataset.
The following animation shows a series of images produced by the *generator* as it was trained for 50 epochs.
 The images begin as random noise, and increasingly resemble hand written digits over time.
# ![sample output](https://tensorflow.org/images/gan/dcgan.gif)
# To learn more about GANs, we recommend MIT's [Intro to Deep Learning](http://introtodeeplearning.com/) course.
"""
import glob
import logging
import os
import time

import PIL
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display

# ### Load and prepare the dataset
# You will use the MNIST dataset to train the generator and the discriminator.
# The generator will generate handwritten digits resembling the MNIST data.

# In[8]:


(TRAIN_IMAGES, TRAIN_LABELS), (_, _) = tf.keras.datasets.mnist.load_data()

# In[9]:


TRAIN_IMAGES = TRAIN_IMAGES.reshape(TRAIN_IMAGES.shape[0], 28, 28, 1).astype("float32")
TRAIN_IMAGES = (TRAIN_IMAGES - 127.5) / 127.5  # Normalize the images to [-1, 1]

# In[10]:


BUFFER_SIZE = 60000
BATCH_SIZE = 256

# In[11]:


# Batch and shuffle the data
TRAIN_DATASET = tf.data.Dataset.from_tensor_slices(TRAIN_IMAGES) \
    .shuffle(BUFFER_SIZE) \
    .batch(BATCH_SIZE)


# In[12]:
# Create the models
# Both the generator and discriminator are defined using the
# [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).


def make_generator_model():
    """
    The Generator
    The generator uses `tf.keras.layers.Conv2DTranspose` (upsampling)
    tf.keras.layers.to produce an image from a seed (random noise).
    Start with a `Dense` layer that takes this seed as input,
    then upsample several times until you reach the desired image size of 28x28x1.
    Notice the `tf.keras.layers.LeakyReLU` activation for each layer, except the output layer which uses tanh.

    :return:
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )
    assert model.output_shape == (None, 28, 28, 1)

    return model


# Use the (as yet untrained) generator to create an image.

# In[13]:


GENERATOR = make_generator_model()

NOISE = tf.random.normal([1, 100])
GENERATED_IMAGE = GENERATOR(NOISE, training=False)

plt.imshow(GENERATED_IMAGE[0, :, :, 0], cmap="gray")


# In[14]:


def make_discriminator_model():
    """
    The Discriminator
    The discriminator is a CNN-based image classifier.

    :return:
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
        )
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


# Use the (as yet untrained) discriminator to classify the generated images as real or fake.
# The model will be trained to output positive values for real images, and negative values for fake images.

# In[15]:


DISCRIMINATOR = make_discriminator_model()
DECISION = DISCRIMINATOR(GENERATED_IMAGE)
print(DECISION)

# ## Define the loss and optimizers
# Define loss functions and optimizers for both models.

# In[16]:


# This method returns a helper function to compute cross entropy loss
CROSS_ENTROPY = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[17]:


def discriminator_loss(real_output, fake_output):
    """
    Discriminator loss
    This method quantifies how well the discriminator is able to distinguish real images from fakes.
    It compares the discriminator's predictions on real images to an array of 1s,
    and the discriminator's predictions on fake (generated) images to an array of 0s.

    :param real_output:
    :param fake_output:
    :return:
    """
    real_loss = CROSS_ENTROPY(tf.ones_like(real_output), real_output)
    fake_loss = CROSS_ENTROPY(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# In[18]:


def generator_loss(fake_output):
    """
    ### Generator loss
    The generator's loss quantifies how well it was able to trick the discriminator.
    Intuitively, if the generator is performing well,
    the discriminator will classify the fake images as real (or 1).
    Here, we will compare the discriminators decisions on the generated images to an array of 1s.

    :param fake_output:
    :return:
    """
    return CROSS_ENTROPY(tf.ones_like(fake_output), fake_output)


# The discriminator and the generator optimizers are different since we will train two networks separately.

# In[19]:


GENERATOR_OPTIMIZER = tf.keras.optimizers.Adam(1e-4)
DISCRIMINATOR_OPTIMIZER = tf.keras.optimizers.Adam(1e-4)

# ### Save checkpoints
# This notebook also demonstrates how to save and restore models,
# which can be helpful in case a long running training task is interrupted.

# In[20]:


CHECKPOINT_DIR = "./training_checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
CHECKPOINT = tf.train.Checkpoint(
    generator_optimizer=GENERATOR_OPTIMIZER,
    discriminator_optimizer=DISCRIMINATOR_OPTIMIZER,
    generator=GENERATOR,
    discriminator=DISCRIMINATOR,
)

# ## Define the training loop
# In[21]:


EPOCHS = 50
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
SEED = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])


# In[22]:


@tf.function
def train_step(images):
    """
    Notice the use of `tf.function`
    This annotation causes the function to be "compiled".
    The training loop begins with generator receiving a random seed as input.
    That seed is used to produce an image.
    The discriminator is then used to classify real images
    (drawn from the training set) and fakes images (produced by the generator).
    The loss is calculated for each of these models,
    and the gradients are used to update the generator and discriminator.

    :param images:
    :return:
    """
    noise_rand = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = GENERATOR(noise_rand, training=True)

        real_output = DISCRIMINATOR(images, training=True)
        fake_output = DISCRIMINATOR(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, GENERATOR.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, DISCRIMINATOR.trainable_variables
    )

    GENERATOR_OPTIMIZER.apply_gradients(
        zip(gradients_of_generator, GENERATOR.trainable_variables)
    )
    DISCRIMINATOR_OPTIMIZER.apply_gradients(
        zip(gradients_of_discriminator, DISCRIMINATOR.trainable_variables)
    )


# In[23]:


def train(dataset, epochs):
    """
    handle epoch logic
    :param dataset:
    :param epochs:
    :return:
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(GENERATOR, epoch + 1, SEED)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            CHECKPOINT.save(file_prefix=CHECKPOINT_PREFIX)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(GENERATOR, epochs, SEED)


# In[24]:


def generate_and_save_images(model, epoch, test_input):
    """
    # Notice `training` is set to False.
    # This is so all tf.keras.layers.run in inference mode (batchnorm).

    # **Generate and save images**

    :param model:
    :param epoch:
    :param test_input:
    :return:
    """
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    logging.debug("%r", "fig.figsize = {}".format(fig.figsize))
    for ind in range(predictions.shape[0]):
        plt.subplot(4, 4, ind + 1)
        plt.imshow(predictions[ind, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


# ## Train the model
# Call the `train()` method defined above to train the generator and discriminator simultaneously.
# Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other
# (e.g., that they train at a similar rate).
# At the beginning of the training,
# the generated images look like random noise.
# As training progresses, the generated digits will look increasingly real.
# After about 50 epochs, they resemble MNIST digits.
# This may take about one minute / epoch with the default settings on Colab.

# In[25]:


train(TRAIN_DATASET, EPOCHS)

# Restore the latest checkpoint.

# In[26]:


CHECKPOINT.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))


# ## Create a GIF

# In[27]:


def display_image(epoch_no):
    """
    # Display a single image using the epoch number
    :param epoch_no:
    :return:
    """
    return PIL.Image.open("image_at_epoch_{:04d}.png".format(epoch_no))


# In[28]:


display_image(EPOCHS)

# Use `imageio` to create an animated gif using the images saved during training.

# In[29]:


ANIM_FILE = "dcgan.gif"

with imageio.get_writer(ANIM_FILE, mode="I") as writer:
    FILENAMES = glob.glob("image*.png")
    FILENAMES = sorted(FILENAMES)
    LAST = -1
    for i, filename in enumerate(FILENAMES):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(LAST):
            LAST = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)

display.Image(filename=ANIM_FILE)

# ## Next steps

# This tutorial has shown the complete code necessary to write and train a GAN.
# As a next step, you might like to experiment with a different dataset,
# for example the Large-scale Celeb Faces Attributes (CelebA) dataset
# [available on Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset/home).
# To learn more about GANs we recommend the [NIPS 2016 Tutorial:
# Generative Adversarial Networks](https://arxiv.org/abs/1701.00160).
