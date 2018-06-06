#! usr/bin/env python3

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import numpy as np

seed = 7
np.random.seed(seed)

# training and testing split
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Flattening the 28 x 28 image into 784 x 0 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],  num_pixels).astype(float)
X_test = X_test.reshape(X_test.shape[0],  num_pixels).astype(float)

# Normalize each value so that it lies in the range 0 - 1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode the o/p
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

# baseline model
def baseline_model():
  model = Sequential()
  model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
  model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model

model = baseline_model()
# Train
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# Evaluate
scores = model.evaluate(X_test, Y_test, verbose=0)

print("Baseline Error: %.2f%%" % (100 - scores[1]*100))