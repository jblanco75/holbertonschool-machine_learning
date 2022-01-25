#!/usr/bin/env python3
"""
Function that builds a neural network with the Keras library
without using Sequential class
"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: is the number of input features to the network
    layers: is a list containing the number of nodes in each
            layer of the network
    activations: is a list containing the activation functions
                 used for each layer of the network
    lambtha: is the L2 regularization parameter
    keep_prob: is the probability that a node will be kept for dropout
    You are not allowed to use the Sequential class
    Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)
    y = inputs
    for layer in range(len(layers)):
        if layer > 0:
            y = K.layers.Dropout(rate=1 - keep_prob)(y)
        y = K.layers.Dense(layers[layer],
                           activation=activations[layer],
                           kernel_regularizer=regularizer)(y)
    return K.Model(inputs=inputs, outputs=y)
