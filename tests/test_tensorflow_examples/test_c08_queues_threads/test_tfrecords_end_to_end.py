# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Thu Jan 26 00:41:43 2017

@author: tomhope
"""

import os
import tensorflow as tf
from keras.datasets.mnist import load_data

#### WRITE TFRECORDS  # noqa
save_dir = "D:\\mnist"


# Download data to save_Dir
def main():
    tf.compat.v1.disable_eager_execution()
    data_sets = load_data(
        path=save_dir,
        # dtype=tf.uint8,
        # reshape=False,
        # validation_size=1000
    )
    data_splits = ["train", "test", "validation"]
    for d in range(len(data_splits)):
        print("saving " + data_splits[d])
        data_set = data_sets[d]

        filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
        writer = tf.io.TFRecordWriter(filename)
        for index in range(data_set.images.shape[0]):
            image = data_set.images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[data_set.images.shape[1]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[data_set.images.shape[2]])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[data_set.images.shape[3]])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[int(data_set.labels[index])])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[image]))}))
            writer.write(example.SerializeToString())
        writer.close()

    # READ
    NUM_EPOCHS = 10

    filename = os.path.join("D:\\mnist", "train.tfrecords")

    filename_queue = tf.compat.v1.train.string_input_producer(
        [filename], num_epochs=NUM_EPOCHS)

    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.io.parse_single_example(
        serialized=serialized_example,
        features={
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })

    image = tf.io.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])

    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    label = tf.cast(features['label'], tf.int32)

    # Shuffle the examples + batch
    images_batch, labels_batch = tf.compat.v1.train.shuffle_batch(
        [image, label], batch_size=128,
        capacity=2000,
        min_after_dequeue=1000)

    W = tf.compat.v1.get_variable("W", [28 * 28, 10])
    y_pred = tf.matmul(images_batch, W)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)

    loss_mean = tf.reduce_mean(input_tensor=loss)

    train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    init = tf.compat.v1.local_variables_initializer()
    sess.run(init)

    # coordinator
    coord = tf.train.Coordinator()
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            sess.run([train_op])
            if step % 500 == 0:
                loss_mean_val = sess.run([loss_mean])
                print(step)
                print(loss_mean_val)
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # example -- get image,label
    img1, lbl1 = sess.run([image, label])

    # example - get random batch
    labels, images = sess.run([labels_batch, images_batch])


if __name__ == "__main__":
    main()
