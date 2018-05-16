#! usr/bin/env python3

# Third Party Imports
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
# Local imports
from utils import gimme_path

def create_model():
      model = Sequential()
      model.add(Dense(12, input_dim=8, activation='relu')) 
      model.add(Dense(8, activation='relu'))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      return model

seed = 7
np.random.seed(seed)

dataset_path = gimme_path('datasets', 'pima-indians-diabetes.csv')

pima = np.loadtxt(dataset_path, delimiter=',', skiprows=9)
X = pima[:,:8]
Y = pima[:,8]

# Build Model
# Keras Classifier, build_fn, takes name of fucntion to create model
# implicitly calls fit()
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

"""
TypeError: can't pickle _thread.RLock objects

From Keras Doc
build_fn is NOT called!! It should construct, compile and return a Keras Model

Related SO Question: https://stackoverflow.com/questions/48303166/keras-typeerror-cant-pickle-thread-lock-objects-with-kerasclassifier
"""