#!/bin/python

import keras

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 4
nb_classes = 10

model = Sequential()

model.add(LSTM(32, return_sequences=True,
              input_shape=(timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics = ['accuracy'])


x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, nb_classes))

x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, nb_classes))

model.fit(x_train, y_train, batch_size=64, nb_epoch=10, validation_data=(x_val,y_val))
