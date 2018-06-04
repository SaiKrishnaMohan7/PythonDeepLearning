#! usr/bin/env python3

# Third Party imports
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelEncoder

# Standard Lib imports
import os

def step_decay(epoch):
  initial_rate = 0.1
  drop = 0.5
  epochs_drop = 10.0
  lr = initial_rate * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
  return lr

# fixed seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset and split as features and labels
ds_path = os.path.join('Datasets', 'ionosphere.csv')
df = read_csv(ds_path, header=None)
ds = df.values
X = ds[:, :34].astype(float)
Y = ds[:, 34]

# Encoding labels
encoder = LabelEncoder()
encoder.fit(Y)
Y_en = encoder.transform(Y)

model = Sequential()

model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))


sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Learning Rate Schedule callback
l_rate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

model.fit(X, Y_en, validation_split=0.33, epochs=50, batch_size=28, callbacks=callbacks_list, verbose=2)