#! usr/bin/env python3

"""
  Helper functions for common tasks script
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
import os

def gimme_path(folder_name, file_name):
  """
  Locates and returns Dataset
  :param: <str> folder_name
  :param: <str> file_name
  :return: <str> file_path
  """
  file_path = os.path.join(folder_name, file_name)
  return file_path


def gimme_network(units, dim, act):
  """
  Given the number of units in a layer returns a network
  :param: <tuple> units, immutable ordered tuple of units/layer
  :param: <int> dim
  :param: <tuple> act, Activation function/layer
  :return: <instance_sequential> model
  """
  # TODO: use for loop to make this better
  model = Sequential([
    Dense(units[0], input_dim=dim),
    Activation(act[0]),
    Dense(units[1]),
    Activation(act[1]),
    Dense(units[2]),
    Activation(act[2])
  ])

  return model
