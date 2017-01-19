
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import os

os.chdir('/home/dldev/magicyang/dl_learn/py')
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 28 * 28 = 784
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y = x*W + b
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder('float', [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  if i % 100 == 0:
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

def weight_variable(shape): # add noises to avoid symmetry and zero gredient
    initial = tf.truncated_normal(shape, stddev=0.1) # has excluded variables more than 2 stddev
    return tf.Variable(initial)

def bias_variable(shape): # use a little positive number to avoid zero constant output(dead neaurons) 
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): # stride of 1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x): # stride of 2
    # ksize [batch, height, width, channels]
    # batch: max_pool over batch examples
    # channel: RGB is 3 and so on 
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1])

# 28 -> 14
# zero padded: 28 -> 32
# conv: 32 -> 28
# pool: 28 -> 14
W_conv1 = weight_variable([5,5,1,32]) # [height, width, input_channel, output_channel]
b_conv1 = bias_variable([32]) # bias number is output_channel

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 14 -> 7
# zero padded: 14 -> 18
# conv: 18 -> 14
# pool: 14 -> 7
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 7*7 full conntected layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# train & validate
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess.run(tf.initialize_all_variables())
for i in xrange(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print 'step %d, training accuracy %g' % (i, train_accuracy)
        print 'test accuracy %g' % accuracy.eval(session=sess, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    train_step.run(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.5})

