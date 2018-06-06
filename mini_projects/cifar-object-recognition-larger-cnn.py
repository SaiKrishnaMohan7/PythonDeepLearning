#! usr/bin/env python3

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np

K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

# load data
(X_train, Y_train),(X_test, Y_test) = cifar10.load_data()

# Normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

def baseline_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu')))
  model.add(Dropout(0.2))
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Dropout(0.2))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dense(num_classes, activation='softmax'))

  return model

model = baseline_model()
epochs = 25
lr = 0.01
decay = lr / epochs
sgd = SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=64)

scores = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))