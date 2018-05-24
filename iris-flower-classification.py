#! usr/bin/env python3

# Third Party imports
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Standard imports
import os

# baseline model model constructed and compiled, assigned to build_fn
def create_model():
  model = Sequential()
  # input layer, 4 features => 4 nodes
  model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
  # output layer, 3 label => 3 nodes
  model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

seed = 7
np.random.seed(seed)

dataset_path = os.path.join('Datasets', 'iris-flower.csv')
# load dataset
df = read_csv(dataset_path, header=None)
dataset = df.values

# features and labels
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as ints
encoder = LabelEncoder()
encoder.fit(Y)
Y_encoded = encoder.transform(Y)
# one hot encoding
Y_hot = np_utils.to_categorical(Y_encoded)

# KerasClassifier
estimator = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)

# Model Evaluation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, Y_hot, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print(f'Accuracy: {results.mean()*100}, Std: {results.std()*100}')