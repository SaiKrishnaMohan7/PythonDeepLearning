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

# Train model
history = model.fit(X, Y, validation_split=0.33,epochs=150, batch_size=10, verbose=0)

# print history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()