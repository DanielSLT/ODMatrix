import numpy as np
import keras
from keras import layers
from keras.engine.topology import Layer
from keras.models import Model
from keras import backend as K
from keras.layers.wrappers import TimeDistributed

K.clear_session()


# define merge layer
class MergeOD(Layer):
    def __init__(self, **kwargs):
        super(MergeOD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.para1 = self.add_weight(shape=(input_shape[0][1], input_shape[0][2], 1),
                                     initializer='uniform', trainable=True,
                                     name='para1')
        super(MergeOD, self).build(input_shape)

    def call(self, inputs):
        mat1 = inputs[0]
        mat2 = inputs[1]
        output = mat1 * self.para1 + mat2
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# number of time stamps
num_cell = 4

# input dimension
img_shape = (num_cell, 285, 285, 1)

# encoding dimension
encoding_dim = 64

# input OD time series
input_OD = keras.Input(shape=img_shape, name='OD_input')

# encoding part
x = TimeDistributed(layers.Conv2D(16, 6,
                                  padding='same',
                                  activation='relu'))(input_OD)

shape_before_Maxpool = K.int_shape(x)

x = TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))(x)
x = TimeDistributed(layers.Conv2D(32, 6,
                                  padding='same',
                                  activation='relu',
                                  strides=(2, 2)))(x)
x = TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))(x)
x = TimeDistributed(layers.Conv2D(4, 6,
                                  padding='same',
                                  activation='relu'))(x)

x = TimeDistributed(layers.Flatten())(x)
encoder_bef_reshape = layers.Dense(encoding_dim, activation='relu')(x)
encoder_output = layers.Reshape((encoding_dim, num_cell, 1), input_shape=(num_cell, encoding_dim))(encoder_bef_reshape)

# input time information
time_input = layers.Input(shape=(1, num_cell, 1), name='time_input')
concat = layers.concatenate([encoder_output, time_input], axis=1)

output_size = encoding_dim

# predicting part
X = layers.Conv2D(64, 2,
                  padding='same',
                  activation='relu')(concat)

X = layers.MaxPooling2D(pool_size=(2, 2),
                        strides=None,
                        padding='same')(X)

X = layers.Conv2D(32, 2,
                  padding='same',
                  activation='relu')(X)
X = layers.MaxPooling2D(pool_size=(2, 2),
                        strides=None,
                        padding='same')(X)

X = layers.Conv2D(16, 2,
                  padding='same',
                  activation='relu')(X)

X = layers.Flatten()(X)
X = layers.Dense(output_size, activation='relu')(X)

x = layers.Dense(np.prod(shape_before_Maxpool[2:]),
                 activation='relu')(X)

x = layers.Reshape(shape_before_Maxpool[2:])(x)

# decoding part
x = layers.Conv2DTranspose(32, 6,
                           padding='same',
                           activation='relu',
                           )(x)
tmp_preOD = layers.Conv2D(1, 6,
                          padding='same',
                          activation='linear', name='tmpODoutput')(x)

# input the output of modified gravity model
reg_OD_input = keras.Input(shape=(285, 285, 1), name='reg_OD_input')

# merge the output of deep learning model and the output of modified gravity model
outputOD = MergeOD()([reg_OD_input, tmp_preOD])

# construct model
finish_model = Model([input_OD, time_input, reg_OD_input], [outputOD])
finish_model.summary()
finish_model.compile(loss='mean_squared_error', optimizer='adam')
