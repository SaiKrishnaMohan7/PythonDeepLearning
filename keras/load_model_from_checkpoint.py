#! usr/bin/env python3

from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import os
import numpy as np

# fixed random seed, for reproducibility
np.random.seed(7)

dataset_path = os.path.join('datasets', 'pima-indians-diabetes.csv')

# load dataset into memory - loads it as 2D numpy array
pima = np.loadtxt(dataset_path, delimiter=',', skiprows=9)

# Split into features and labels (i/p and o/p)
X = pima[:,:8]
Y = pima[:,8]

# load model architecture from memory
with open(os.path.join('keras', 'model_archs','load_from_checkpoint_model_arch.yaml'), 'r') as model_arch_file:
  loaded_model_yaml = model_arch_file.read()

loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights(os.path.join('keras', 'model_checkpoint_only_best', 'weights.best.hdf5'))

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print('loaded model arch and weights from checkpoint')

print(f'\n {loaded_model.metrics_names[1], score[1]*100}')