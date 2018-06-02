#! usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

seed = 7

# fixed random seed, for reproducibility
np.random.seed(seed)

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
model.add(Dense(12, input_dim=8, kernel_initializer='uniform',activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the model for efficient computation.
# The backend, TensorFlow default, chooses the most efficient way to represent the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Checkpoint
# This filename will save model weights everytime keras thinks that there's an improvement
# instead, setting the below var to 'weights.best.hdf5' will save ONLY the BEST and to the same file!
# runnign with the below config gave me 24 files
filepath = 'weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
cb_list = [checkpoint]

# Train model
model.fit(X, Y, validation_split=0.33,epochs=150, batch_size=10, callbacks=cb_list, verbose=0)

# Save model arch as checkpoints only save weights
model_arch = model.to_yaml()
with open('load_from_checkpoint_model_arch.yaml', 'w') as arch_yaml:
  arch_yaml.write(model_arch)