#!/usr/bin/env python3
"""
Function that builds a transition layer as described
in Densely Connected Convolutional Networks
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of
    filters within the output, respectively
    """
    w = K.initializers.HeNormal()
    BN = K.layers.BatchNormalization()(X)
    act_BN = K.layers.Activation('relu')(BN)
    Tr_filters = int(nb_filters * compression)
    Tr_layer = K.layers.Conv2D(filters=Tr_filters,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=w)(act_BN)
    Tr_pooling = K.layers.AveragePooling2D(pool_size=2,
                                           strides=2)(Tr_layer)
    return Tr_pooling, Tr_filters
