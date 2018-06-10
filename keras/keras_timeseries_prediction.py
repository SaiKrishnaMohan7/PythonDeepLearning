#! usr/bin/env python3

from pandas import read_csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

import string

seed = 7
np.random.seed(seed)

alphabets = string.ascii_uppercase
# map each alphabet to a integer position
int_maps = dict((c, i) for i, c in enumerate(alphabets))
# map each alphabet to int for converting back
alpha_maps = dict((i, c) for i, c in enumerate(alphabets))

# Prep Datasetm mapping all the chars to integers
seq = 3
dataX = []
dataY = []

for i in range(0, len(alphabets) - seq, 1):
  seq_in = alphabets[i: i + seq]
  seq_out = alphabets[i + seq]
  dataX.append([int_maps[char] for char in seq_in])
  dataY.append(int_maps[seq_out])
  print(seq_in, '->', seq_out)

# Reshaping data to a format acceptable by LSTM networks [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), 1, seq))
# normalize input
X = X / float(len(alphabets))

# one ht encode
Y = np_utils.to_categorical(dataY)

# create and fit n/w
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=500,batch_size=1, verbose=2)

# summarize performance of the model
scores = model.evaluate(X, Y, verbose=0)
print('Model Accuracy: %.2f%%' % (scores[1]*100))

for pattern in dataX:
  x = np.reshape(pattern,(1, 1,len(pattern)))
  x = x / float(len(alphabets))
  prediction = model.predict(x, verbose=0)
  index = np.argmax(prediction)
  result = alpha_maps[index]
  seq_in = [alpha_maps[value] for value in pattern]
  print(seq_in, '->', result)