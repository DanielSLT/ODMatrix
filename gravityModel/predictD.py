import keras
from keras import layers
from keras.models import Model

# number of time stamps
num_cell = 4

# input dimension
input_shape = (285, num_cell, 1)

# output dimension
output_size = 285

# input D time series
input_x = keras.Input(shape=input_shape)

# define model
X = layers.Conv2D(128, 2,
                  padding='same',
                  activation='relu')(input_x)

X = layers.MaxPooling2D(pool_size=(2, 2),
                        strides=None,
                        padding='same')(X)

X = layers.Conv2D(64, 2,
                  padding='same',
                  activation='relu')(X)

X = layers.MaxPooling2D(pool_size=(2, 2),
                        strides=None,
                        padding='same')(X)

X = layers.Conv2D(64, 2,
                  padding='same',
                  activation='relu')(X)
X = layers.MaxPooling2D(pool_size=(2, 2),
                        strides=None,
                        padding='same')(X)
X = layers.Conv2D(32, 2,
                  padding='same',
                  activation='relu')(X)
X = layers.MaxPooling2D(pool_size=(2, 2),
                        strides=None,
                        padding='same')(X)
X = layers.Flatten()(X)
x = layers.Dense(output_size, activation='relu')(X)

# construct model
predictModel = Model(input_x, x)
print(predictModel.summary())

predictModel.compile(loss='mean_squared_error', optimizer='adam')
