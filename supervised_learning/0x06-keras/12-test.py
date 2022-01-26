#!/usr/bin/env python3
"""
Function that tests a neural network
"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Returns: the loss and accuracy of the model
    with the testing data, respectively
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
