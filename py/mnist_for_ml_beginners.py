#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:31:29 2017

@author: magicyang
"""
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# 28 * 28 = 784
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder('float', [None,10])

    # y = x*W + b
y = tf.nn.softmax(tf.matmul(x,W) + b)
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
  print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
