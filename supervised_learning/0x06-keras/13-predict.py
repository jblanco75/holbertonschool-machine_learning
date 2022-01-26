#!/usr/bin/env python3
"""
Function that makes a prediction using a neural network
"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Returns: the prediction for the data
    """
    return network.predict(data, verbose=verbose)
