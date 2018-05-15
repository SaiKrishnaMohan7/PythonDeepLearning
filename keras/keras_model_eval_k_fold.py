#! usr/bin/env python3
"""
  Usign k-fold cross-validation to evaluate performance
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import StratifiedKFold
from utils import gimme_network, gimme_path
import numpy as np
import os

seed = 7
np.random.seed(seed)

dataset_path = gimme_path('datasets', 'pima-indians-diabetes.csv')
pima = np.loadtxt(dataset_path, delimiter=',', skiprows=9)
X = pima[:,:8]
Y = pima[:,8]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv_scores = list()

for train, test in kfold.split(X, Y):
  model = gimme_network((12, 8, 1), 8, ('relu', 'relu', 'sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  # fit/train model
  model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
  scores = model.evaluate(X[test], Y[test], verbose=0)

  print(f'{model.metrics_names[1]} and {scores[1]*100}')

cv_scores.append(scores[1]*100)
print(f'mean accuracy: {np.mean(cv_scores)}\n spread: {np.std(cv_scores)}')