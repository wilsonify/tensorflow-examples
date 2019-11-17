#!/usr/bin/env python
# coding: utf-8

# # Chapter 10

# ## Assigning loaded ​weights - Single weight array

# ### Saving the weights of the trained model

# In[ ]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100


data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true))


gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:

    # Train
    sess.run(tf.global_variables_initializer())

    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    weights = sess.run(W)


# In[ ]:


import numpy as np
import os 
path = 'tmp//'

np.savez(os.path.join(path, 'weight_storage'), weights)


# ### Load weights

# In[ ]:


loaded_w = np.load(path + 'weight_storage.npz')
loaded_w = loaded_w.items()[0][1]

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # assigning loaded weights
    sess.run(W.assign(loaded_w))
    acc = sess.run(accuracy, feed_dict={x: data.test.images, 
                                        y_true: data.test.labels})

print("Accuracy: {}".format(acc))


# ### Saving multiple weight arrays

# In[ ]:


class simple_cnn:
    def __init__(self, x_image,keep_prob, weights=None, sess=None):
        
        self.parameters = []
        self.x_image = x_image

        conv1 = self.conv_layer(x_image, shape=[5, 5, 1, 32])
        conv1_pool = self.max_pool_2x2(conv1)

        conv2 = self.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = self.max_pool_2x2(conv2)

        conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
        full_1 = tf.nn.relu(self.full_layer(conv2_flat, 1024))

        full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

        self.y_conv = self.full_layer(full1_drop, 10)
        
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
            
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial,name='weights')


    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,name='biases')


    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], 
                                       padding='SAME')


    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    def conv_layer(self,input, shape):
        W = self.weight_variable(shape)
        b = self.bias_variable([shape[3]])
        self.parameters += [W, b]

        return tf.nn.relu(self.conv2d(input, W) + b)


    def full_layer(self,input, size):
        in_size = int(input.get_shape()[1])
        W = self.weight_variable([in_size, size])
        b = self.bias_variable([size])
        self.parameters += [W, b]
        return tf.matmul(input, W) + b
    

    def load_weights(self, weights, sess):
        for i,w in enumerate(weights):
            print("Weight index: {}".format(i), 
                               "Weight shape: {}".format(w.shape))
            sess.run(self.parameters[i].assign(w))


# In[ ]:


NUM_STEPS = 5000

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()

cnn = simple_cnn(x_image,keep_prob, sess)

cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                                                 logits=cnn.y_conv, labels= y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(cnn.y_conv, 1), 
                                      tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
X = data.test.images.reshape(10, 1000, 784)
Y = data.test.labels.reshape(10, 1000, 10)




for _ in range(NUM_STEPS):
    batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys ,keep_prob:1.0})

test_accuracy = np.mean([sess.run(accuracy, 
                         feed_dict={x:X[i], y_:Y[i],keep_prob:1.0}) 
                         for i in range(10)])    



path = 'tmp//'
weights = sess.run(cnn.parameters)
np.savez(os.path.join(path, 'cnn_weight_storage'), weights)

sess.close()

print("test accuracy: {}".format(test_accuracy))


# ### Loading weights

# In[ ]:



x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()

weights = np.load(path +'cnn_weight_storage.npz')
weights = weights.items()[0][1]
cnn = simple_cnn(x_image,keep_prob,weights, sess)

cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                                                 logits=cnn.y_conv, labels= y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(cnn.y_conv, 1), 
                                      tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

X = data.test.images.reshape(10, 1000, 784)
Y = data.test.labels.reshape(10, 1000, 10)
test_accuracy = np.mean([sess.run(accuracy, 
                         feed_dict={x:X[i], y_:Y[i],keep_prob:1.0}) 
                         for i in range(10)])

sess.close()

print("test accuracy: {}".format(test_accuracy))


# ### Saver class - save and restore weights only

# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
DATA_DIR = '/tmp/data'
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

NUM_STEPS = 1000
MINIBATCH_SIZE = 100

DIR = 'saved_model/'

x = tf.placeholder(tf.float32, [None, 784],name='x')
W = tf.Variable(tf.zeros([784, 10]),name='W')
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

saver = tf.train.Saver(max_to_keep=7, 
                       keep_checkpoint_every_n_hours=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,NUM_STEPS+1):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})
        
        if step % 50 == 0:
            saver.save(sess, os.path.join(DIR, "model_ckpt"), 
                                           global_step=step)
    
    ans = sess.run(accuracy, feed_dict={x: data.test.images, 
                                        y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))


# In[ ]:


tf.reset_default_graph() 
x = tf.placeholder(tf.float32, [None, 784],name='x')
W = tf.Variable(tf.zeros([784, 10]),name='W')
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(DIR,"model_ckpt-1000"))
    ans = sess.run(accuracy, feed_dict={x: data.test.images, 
                                        y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))


# ### Save and restore weights + graph + collections

# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
DATA_DIR = '/tmp/data'
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

NUM_STEPS = 1000
MINIBATCH_SIZE = 100

DIR = 'saved_model/'

x = tf.placeholder(tf.float32, [None, 784],name='x')
W = tf.Variable(tf.zeros([784, 10]),name='W')
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

saver = tf.train.Saver(max_to_keep=7, 
                       keep_checkpoint_every_n_hours=1)

train_var = [x,y_true,accuracy]
tf.add_to_collection('train_var', train_var[0])
tf.add_to_collection('train_var', train_var[1])
tf.add_to_collection('train_var', train_var[2])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,NUM_STEPS+1):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})
        
        if step % 50 == 0:
            saver.export_meta_graph(os.path.join(DIR,"model_ckpt.meta"),collection_list=['train_var'])
            saver.save(sess, os.path.join(DIR, "model_ckpt"), 
                                           global_step=step)
    
    ans = sess.run(accuracy, feed_dict={x: data.test.images, 
                                        y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))


# In[ ]:


tf.reset_default_graph() 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph(os.path.join(DIR,"model_ckpt.meta"))
    saver.restore(sess,  os.path.join(DIR,"model_ckpt-1000"))
    x =  tf.get_collection('train_var')[0]
    y_true =  tf.get_collection('train_var')[1]
    accuracy =  tf.get_collection('train_var')[2]

    ans = sess.run(accuracy, feed_dict={x: data.test.images, 
                                        y_true: data.test.labels})
print("Accuracy: {:.4}%".format(ans*100))


# ### TensorFlow Serving

# In[ ]:


import os
import sys
import tensorflow as tf
from tensorflow.python.saved_model import builder 
                                           as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow_serving.example import mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 10,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer(
                 'model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,dtype='float')

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,dtype='float')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: mnist_export.py [--training_iteration=x] '
              '[--model_version=y] export_dir')
        sys.exit(-1)
    if FLAGS.training_iteration <= 0:
        print('Please specify a positive 
                                 value for training iteration.')
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print ('Please specify a positive 
                                     value for version number.')
        sys.exit(-1)
    

    print('Training...')
    mnist = mnist_input_data.read_data_sets(
                                  FLAGS.work_dir, one_hot=True)
    sess = tf.InteractiveSession()
    serialized_tf_example = tf.placeholder(
                                  tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[784],\ 
                                            dtype=tf.float32),}
    tf_example = tf.parse_example(serialized_tf_example, 
                                                feature_configs)
    
    
    x = tf.identity(tf_example['x'], name='x')  
    y_ = tf.placeholder('float', shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
 
    y = tf.nn.softmax(y_conv, name='y')
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).                                          minimize(cross_entropy)
   
    
    values, indices = tf.nn.top_k(y_conv, 10)
    prediction_classes = tf.contrib.lookup.index_to_string(
      tf.to_int64(indices), 
      mapping=tf.constant([str(i) for i in xrange(10)]))
    
    sess.run(tf.global_variables_initializer())

    for _ in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        
        train_step.run(feed_dict={x: batch[0], 
                                 y_: batch[1], keep_prob: 0.5})
        print(_)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), 
                                         tf.argmax(y_,1))

    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
                       y_: mnist.test.labels})
    
    print('training accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, 
        y_: mnist.test.labels, keep_prob: 1.0}))

    print('training is finished!')
    
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
    print 'Exporting trained model to', export_path
    builder = saved_model_builder.SavedModelBuilder(export_path)

    classification_inputs = utils.build_tensor_info(
                                             serialized_tf_example)
    classification_outputs_classes = utils.build_tensor_info(
                                             prediction_classes)
    classification_outputs_scores = utils.build_tensor_info(values)

    classification_signature = signature_def_utils.      build_signature_def(
      inputs={signature_constants.CLASSIFY_INPUTS:\ 
                           classification_inputs},
      outputs={
          signature_constants.CLASSIFY_OUTPUT_CLASSES:
              classification_outputs_classes,
          signature_constants.CLASSIFY_OUTPUT_SCORES:
              classification_outputs_scores
      },
      method_name=signature_constants.CLASSIFY_METHOD_NAME)

    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_y = utils.build_tensor_info(y_conv)

    prediction_signature = signature_def_utils.build_signature_def(
      inputs={'images': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.initialize_all_tables(), 
                                   name='legacy_init_op')
    builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      legacy_init_op=legacy_init_op)

    builder.save()

    print('new model exported!')


if __name__ == '__main__':
    tf.app.run()

