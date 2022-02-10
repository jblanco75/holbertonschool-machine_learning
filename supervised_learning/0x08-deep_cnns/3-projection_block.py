#!/usr/bin/env python3
"""
Function that builds a projection block as described in
Deep Residual Learning for Image Recognition (2015)
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
      F11 is the number of filters in the first 1x1 convolution
      F3 is the number of filters in the 3x3 convolution
      F12 is the number of filters in the second 1x1 convolution as well as
      the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution in both the main path and the
    shortcut connection
    All convolutions inside the block should be followed by batch normalization
    along the channels axis and a rectified linear activation (ReLU),
    respectively.
    All weights should use he normal initialization
    Returns: the activated output of the projection block
    """
    w = K.initializers.HeNormal()
    F11, F3, F12 = filters
    conv_F11 = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               padding="same",
                               strides=s,
                               kernel_initializer=w)(A_prev)
    BNF11 = K.layers.BatchNormalization()(conv_F11)
    activ_F11 = K.layers.Activation('relu')(BNF11)
    conv_F3 = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding="same",
                              kernel_initializer=w)(activ_F11)
    BNF3 = K.layers.BatchNormalization()(conv_F3)
    activ_F3 = K.layers.Activation('relu')(BNF3)
    conv_F12 = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding="same",
                               kernel_initializer=w)(activ_F3)
    BNF12 = K.layers.BatchNormalization()(conv_F12)

    conv_SC = K.layers.Conv2D(filters=F12,
                              kernel_size=(1, 1),
                              padding="same",
                              strides=s,
                              kernel_initializer=w)(A_prev)
    BNFSC = K.layers.BatchNormalization()(conv_SC)

    output = K.layers.Add()([BNF12, BNFSC])
    activated_output = K.layers.Activation('relu')(output)
    return activated_output
