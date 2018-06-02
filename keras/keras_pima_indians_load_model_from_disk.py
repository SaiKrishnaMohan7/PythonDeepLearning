#! usr/bin/env python3

from keras.models import Sequential
from keras.models import model_from_json
# For loading from YAMl files
# from keras.models import model_from_yaml
from keras.layers import Dense
import os
import numpy as np

# fixed random seed, for reproducibility
np.random.seed(7)

# Ensure code is able to find file on all OSs
dataset_path = os.path.join('datasets', 'pima-indians-diabetes.csv')

# load dataset into memory - loads it as 2D numpy array
pima = np.loadtxt(dataset_path, delimiter=',', skiprows=9)

# Split into features and labels (i/p and o/p)
X = pima[:,:8]
Y = pima[:,8]

# load model architecture from memory
with open(os.path.join('keras', 'pima_model_arch.json'), 'r') as json_file:
  loaded_model_json = json_file.read()

# load model architecture from memory
# with open('pima_model_arch.yaml', 'r') as yaml_file:
#   loaded_model_yaml = yaml_file.read()

loaded_model = model_from_json(loaded_model_json)

# loaded_model = model_from_yaml(loaded_model_yaml)

# load weights into new model
loaded_model.load_weights(os.path.join('keras', 'pima_model_weights.h5'))
print('loaded model from disk')

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)

print(f'\n {loaded_model.metrics_names[1], score[1]*100}')