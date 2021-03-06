#! usr/bin/env python3

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np
# Theano?
K.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)

# training and testing split
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape to samples, channels, width, height
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype(float)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype(float)

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
  # Convolutional Layer
  model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
  # MaxPooling Layer
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Dropout/Regularization layer
  model.add(Dropout(0.2))
  # Flattening layer for fully connected layer
  model.add(Flatten())
  # Fully connected layer
  model.add(Dense(128, activation='relu'))
  # Output Layer
  model.add(Dense(10, activation='softmax'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model

model = baseline_model()
# Train
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# Evaluate
scores = model.evaluate(X_test, Y_test, verbose=0)

print("CNN Error: %.2f%%" % (100 - scores[1]*100))