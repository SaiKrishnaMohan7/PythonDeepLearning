#! usr/bin/env python3

# Third Party imports
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

# Standard Lib imports
import os

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

# Learning Rate calculation
epochs = 50
lr = 0.1
decay_rate = lr / epochs
momentum = 0.8

sgd = SGD(lr=lr, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X, Y_en, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2)