#! usr/bin/env python3

from keras.models import Sequential
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
X = pima[:,:8] # All rows, all columns except 8th, 768 X 7 matrix,, features
Y = pima[:,8] # All rows, just the 8th column , 768 X 1, labels
# print(pima)

# Create Model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(type(model))

# Compile the model for efficient computation.
# The backend, TensorFlow default, chooses the most efficient way to represent the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate model
scores = model.evaluate(X, Y)

print(f'\n {model.metrics_names[1], scores[1]*100}')

# Serialize model to JSON - Saving model arch into json file for later use
model_arch = model.to_json()
# Serialize model to YAML
# model_arch = model.to_yaml()

# Writing the model arch into a new file
with open('pima_model_arch.json', 'w') as json_file:
	json_file.write(model_arch)

# with open('pima_model_arch.json', 'w') as yaml_file:
# 	yaml_file.write(model_arch)

# Serialize the wieghts to HDF5
model.save_weights('pima_model_weights')
print('Saved model weights to disk')