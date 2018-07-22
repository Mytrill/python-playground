# See https://elitedatascience.com/keras-tutorial-deep-learning-in-python

import numpy as np
# Sequential model: linear stack of layers
from keras.models import Sequential
# "core" layers from keras
from keras.layers import Dense, Dropout, Activation, Flatten
# CNN layers from keras
from keras.layers import Convolution2D, MaxPooling2D
# Some utilities
from keras.utils import np_utils
# MNIST dataset
from keras.datasets import mnist

# Set seed for reproducibility
np.random.seed(123)

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# (60000, 28, 28) -> 60  28x28 images
print(X_train.shape)

#
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# (60000, 1, 28, 28)
print X_train.shape