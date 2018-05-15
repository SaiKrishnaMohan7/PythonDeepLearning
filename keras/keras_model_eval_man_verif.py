#! usr/bin/env python3
"""
  Usign Manual Verification to evaluate performance
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from utils import gimme_network, gimme_path
from sklearn.model_selection import train_test_split
import numpy as np
import os

seed = 7
np.random.seed(seed)

dataset_path = gimme_path('datasets', 'pima-indians-diabetes.csv')
pima = np.loadtxt(dataset_path, delimiter=',', skiprows=9)
model = gimme_network((12, 8, 1), 8, ('relu', 'relu', 'sigmoid'))
X = pima[:,:8]
Y = pima[:,8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit/train model spilt 33% of the data for validating
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150, batch_size=10)