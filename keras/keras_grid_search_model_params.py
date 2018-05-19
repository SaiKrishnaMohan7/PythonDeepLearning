#! usr/bin/env python3

# Third Party Imports
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
# Local imports
from utils import gimme_path

def create_model(optimizer='rmsprop', init='glorot_uniform'):
      model = Sequential()
      model.add(Dense(12, input_dim=8, kernel_initializer=init,activation='relu')) 
      model.add(Dense(8, kernel_initializer=init, activation='relu'))
      model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
      return model

seed = 7
np.random.seed(seed)

dataset_path = gimme_path('datasets', 'pima-indians-diabetes.csv')

pima = np.loadtxt(dataset_path, delimiter=',', skiprows=9)
X = pima[:,:8]
Y = pima[:,8]


model = KerasClassifier(build_fn=create_model, verbose=0)

optimizers = ['rmsprop', 'adam']
inits = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)

print(f'Best: {grid_result.best_score_} using, {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, std, param in zip(means, stds, params):
  print(f'Mean: {mean}, Deviation: {std} with param')