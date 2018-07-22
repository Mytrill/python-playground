import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Generate dummy data
data = data = np.linspace(1, 2, 100).reshape(-1, 1)
y = data * 5

# Define the model


def baseline_model():
    model = Sequential()
    model.add(Dense(1, activation='linear', input_dim=1))
    model.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


# Use the model
regr = baseline_model()
regr.fit(data, y, epochs=200, batch_size=32)

plt.

plot(data, regr.predict(data), 'b', data, y, 'k.')
