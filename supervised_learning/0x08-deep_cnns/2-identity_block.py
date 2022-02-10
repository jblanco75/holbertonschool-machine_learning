#!/usr/bin/env python3
"""
Function that builds an identity block as described
in Deep Residual Learning for Image Recognition (2015)
"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
      F11 is the number of filters in the first 1x1 convolution
      F3 is the number of filters in the 3x3 convolution
      F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the identity block
    """
    w = K.initializers.HeNormal()
    F11, F3, F12 = filters
    conv_F11 = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=w)(A_prev)
    BNF11 = K.layers.BatchNormalization(axis=3)(conv_F11)
    activ_F11 = K.layers.Activation('relu')(BNF11)
    conv_F3 = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding='same',
                              kernel_initializer=w)(activ_F11)
    BNF3 = K.layers.BatchNormalization(axis=3)(conv_F3)
    activ_F3 = K.layers.Activation('relu')(BNF3)
    conv_F12 = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=w)(activ_F3)
    BNF12 = K.layers.BatchNormalization(axis=3)(conv_F12)
    activ_F3 = K.layers.Activation('relu')(BNF12)
    BNF = K.layers.BatchNormalization(axis=3)(activ_F3)

    activated_output = K.layers.Add()([BNF, A_prev])
    return activated_output
