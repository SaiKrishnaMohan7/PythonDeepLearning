#! usr/bin/env python3

# Third Party imports
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Standard Lib imports
import os

# fixed seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset and split as features and labels
ds_path = os.path.join('Datasets', 'boston-housing.csv')
df = read_csv(ds_path, delim_whitespace=True, header=None)
ds = df.values
X = ds[:, :13].astype(float)
Y = ds[:, 13]

def create_model():
  # create model
  model = Sequential()
  # adding another hidden layer
  model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model 

# trying out different network topologies
def larger_model():
    # create model
  model = Sequential()
  # adding another hidden layer
  model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
  model.add(Dense(6, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model 


estimator = KerasRegressor(build_fn=larger_model, epochs=100, batch_size=5, verbose=0)
estimators = []
estimators.append(('standardized', StandardScaler()))
estimators.append(('mlp', estimator))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))