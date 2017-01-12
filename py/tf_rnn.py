#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:44:26 2016

@author: magicyang
"""

import tensorflow as tf
import numpy as np

# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
  int2binary[i] = binary[i] # 得到每个数字的bit位表示

import numpy as np
np.random.seed(0)

def encode(a_int):
  return int2binary[a_int]

def decode(a_code):
  return np.packbits(a_code)[0]

# generating data
NUM_EXAMPLES = 1000
binary_dim = 8
largest_number = pow(2, binary_dim)

train_input = []
train_input2 = []

train_output = []

for i in xrange(NUM_EXAMPLES*11):
  a_int = np.random.randint(largest_number/2)
  b_int = np.random.randint(largest_number/2)
  a_code = encode(a_int)
  b_code = encode(b_int)
  c_code = encode(a_int + b_int)
  
  train_input.append(zip(a_code,b_code))
  
  train_output.append(c_code)
  
test_input = train_input[NUM_EXAMPLES:] 
test_output = train_output[NUM_EXAMPLES:] #everything beyond 10,000

train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]


# import packages
import tensorflow as tf


# First phase is building the computation graph where you define all the
# calculations and functions

# define two varibles which will hold the input data and the target data
# dimensions for data [Batch Size, Sequence Length, Input Dimension]
# Batch Size: None unknown and to be determined at runtime
# placeholders will be supplied with data later
data = tf.placeholder(tf.float32, [None, binary_dim, 2])
target = tf.placeholder(tf.float32, [None, binary_dim])

# create RNN cell. 
num_hidden = 16
cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)

# second phase is the execution phase where a Tensorflow session is created
# and the graph defined earlier is executed

# unroll the network and pass the data and store the output in val
# discard the state because every time we lokk at a new sequence, the state becomes irrelevant
# still desgining the model and doesn't mean it is executed
val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

# 将结果转置从而得到sequence最后一个
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

# last*weight+bias 得到对于所有类别值的矩阵, 计算softmax得到概率分
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

# loss function
# clip_by_value(val, min, max): bounding
#cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
l1_loss = -tf.reduce_mean(target - tf.clip_by_value(prediction,1e-10,1.0))

# 设置优化算法
optimizer = tf.train.AdamOptimizer()
#minimize = optimizer.minimize(cross_entropy)
minimize = optimizer.minimize(l1_loss)


# 计算测试集指标
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


# Execution

# 初始化所有定义的变量并创建session
# 前面定义各种函数指针，session调用函数
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 1000
num_of_batches = int(len(train_input) / batch_size)
epoch = 1000
for i in range(epoch):
  ptr = 0
  for j in range(num_of_batches):
    inp = train_input[ptr:ptr+batch_size]
    out = train_output[ptr:ptr+batch_size]
    ptr += batch_size
    sess.run(minimize, {data:inp, target: out})
  print 'Epoch - ', str(i)
  incorrect = sess.run(error, {data:test_input, target:test_output})  
  print('Epoch {:2d} error {:3.1f}%'.format(i+1, 100*incorrect))

sess.close()


























