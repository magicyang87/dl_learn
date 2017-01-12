#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:46:03 2017

@author: magicyang
"""

import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2, 100)) # random input
y_data = np.dot([0.1, 0.2], x_data) + 0.3   # true y


# learn y
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b  # define y

# define optimize method
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(0, 201):
  sess.run(train)
  if step % 20 == 0:
    print step, sess.run(W), sess.run(b)
    