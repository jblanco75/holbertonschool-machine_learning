#!/usr/bin/env python3
"""
Function that builds a dense block as described
in Densely Connected Convolutional Networks
"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense
    Block and the number of filters within the concatenated
    outputs, respectively
    """
    w = K.initializers.HeNormal()
    for layer in range(layers):
        BN_1 = K.layers.BatchNormalization()(X)
        act_1 = K.layers.Activation('relu')(BN_1)
        conv_1 = K.layers.Conv2D(filters=(4 * growth_rate),
                                 kernel_size=(1, 1),
                                 padding='same',
                                 kernel_initializer=w)(act_1)
        BN_2 = K.layers.BatchNormalization()(conv_1)
        act_2 = K.layers.Activation('relu')(BN_2)
        conv_2 = K.layers.Conv2D(filters=growth_rate,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 kernel_initializer=w)(act_2)
        X = K.layers.concatenate([X, conv_2])
        nb_filters += growth_rate
    return X, nb_filters
