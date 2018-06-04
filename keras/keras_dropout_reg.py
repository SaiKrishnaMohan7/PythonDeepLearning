#! usr/bin/env python3

# Third Party imports
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
# for calculating norm of a vector
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Standard Lib imports
import os

# fixed seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset and split as features and labels
ds_path = os.path.join('Datasets', 'sonar.csv')
df = read_csv(ds_path, header=None)
ds = df.values
X = ds[:, :60].astype(float)
Y = ds[:, 60]

# Encoding labels
encoder = LabelEncoder()
encoder.fit(Y)
Y_en = encoder.transform(Y)

def create_model():
  # create model
  model = Sequential()
  # adding another hidden layer
  model.add(Dropout(0.2, input_shape=(60,)))
  model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.2))
  model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(0.2)))
  model.add(Dropout(0.2))
  model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
  sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
  model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
  return model

# baseline model evaluation with stanardized dataset
estimator = KerasClassifier(build_fn=create_model, epochs=300, batch_size=5, verbose=0)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', estimator))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, Y_en, cv=kfold)
results = cross_val_score(pipeline, X, Y_en, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))