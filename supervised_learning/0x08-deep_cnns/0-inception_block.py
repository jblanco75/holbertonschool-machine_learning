#!/usr/bin/env python3
"""
Function that builds an inception block as
described in Going Deeper with Convolutions (2014)
"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
      F1 is the number of filters in the 1x1 convolution
      F3R is the number of filters in the 1x1 convolution
      before the 3x3 convolution
      F3 is the number of filters in the 3x3 convolution
      F5R is the number of filters in the 1x1 convolution
      before the 5x5 convolution
      F5 is the number of filters in the 5x5 convolution
      FPP is the number of filters in the 1x1 convolution after the
      max pooling (Note : The output shape after the max pooling layer
      is outputshape = math.floor((inputshape - 1) / strides) + 1)
    All convolutions inside the inception block should use a rectified
    linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    w = K.initializers.HeNormal()
    F1, F3R, F3, F5R, F5, FPP = filters
    conv_F1 = K.layers.Conv2D(filters=F1,
                              kernel_size=(1, 1),
                              padding='same',
                              activation='relu',
                              kernel_initializer=w)(A_prev)
    conv_F3R = K.layers.Conv2D(filters=F3R,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=w)(A_prev)
    conv_F3 = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu',
                              kernel_initializer=w)(conv_F3R)
    conv_F5R = K.layers.Conv2D(filters=F5R,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=w)(A_prev)
    conv_F5 = K.layers.Conv2D(filters=F5,
                              kernel_size=(5, 5),
                              padding='same',
                              activation='relu',
                              kernel_initializer=w)(conv_F5R)
    max_pool = K.layers.MaxPooling2D(pool_size=3, strides=1,
                                     padding='same')(A_prev)

    conv_FPP = K.layers.Conv2D(filters=FPP,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=w)(max_pool)

    incept_block = K.layers.concatenate([conv_F1, conv_F3, conv_F5, conv_FPP])
    return incept_block
