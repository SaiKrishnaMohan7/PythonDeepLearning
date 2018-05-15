# Keras Notes

* Lightweight API, provides an interface to interact with Tensorflow or Theano (not sure)

* `p3 -c "from keras import backend; print(backend._BACKEND)"` can be used to check what the backend of kears is configured for. Default config is for tensorflow

* The *backend* to Keras can be configured by setting 
  *KERAS_BACKEND* env var to the preferred backend, say theano, `KERAS_BACKEND=theano`. To check run step 2

* The input layer, in a Sequential Model, requires the shape (ex: input_shape(768, 8)) of the input to be defined (Keras Docs is a good source), all the following layers have automatic shape inference (saves a lot of work)

* Dense is a 2D layer and can accept `input_dim=768`

* Alternately, the Model Definition in keras_pima_indians_model.py could defined as

  ```python
  model = Sequential([
    Dense(12, input_shape=(768, 8)),
    Activation('relu'),
    Dense(12),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid')
])
  ```

    the above snippet assumes

  ```python
  from keras.models import Sequential
  from kears.layers import Dense, Activation
  ```

* Rectifier activation function `f(x) = log(1 + exp x)`, [this](https://en.wikipedia.org/wiki/Rectifier_(neural_networks) "Wikipedia") is a good source.

* `binary_crossentropy` loss function is a logarithmic loss function as opposed to a squared error loss function. [This](http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cost-function "Read The Docs") is an imporving source but looks good, everything in one place and a chance to contribute.

## Problems Encountered

* Not able to import files from directory to another (haven't found a solution yet)