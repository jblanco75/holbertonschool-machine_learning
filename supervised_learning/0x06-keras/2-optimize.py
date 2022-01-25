#!/usr/bin/env python3
"""
Function that sets up Adam optimization for a
keras model with categorical crossentropy loss and accuracy metrics
"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Returns: None
    """
    optimizer = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics='accuracy')
