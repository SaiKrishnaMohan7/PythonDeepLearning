#! usr/bin/env python3
"""
  Usign Keras Automatic Verification to evaluate performance
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from utilities.utils import gimme_network, gimme_path
import numpy as np
import os

np.random.seed(7)

dataset_path = gimme_path('datasets', 'pima-indians-diabetes.csv')
pima = np.loadtxt(dataset_path, delimiter=',', skiprows=9)
model = gimme_network((12, 8, 1), 8, ('relu', 'relu', 'sigmoid'))
X = pima[:,:8]
Y = pima[:,8]

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit/train model spilt 33% of the data for validating
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)