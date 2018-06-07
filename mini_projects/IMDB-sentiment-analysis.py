#! usr/bin/env python3
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import Flatten
from keras.preprocessing import sequence

seed = 7
np.random.seed(seed)

# Set the Vocab list. Just load vocab number words zero rest
vocab = 5000
(X_train, Y_train),(X_test, Y_test) = imdb.load_data(num_words=vocab)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
model.add(Embedding(vocab, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2, batch_size=128, verbose=2)
scores = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))