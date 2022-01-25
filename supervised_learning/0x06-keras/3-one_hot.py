#!/usr/bin/env python3
"""
Function that converts a label vector into a one-hot matrix
"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector to a one-hot matrix"""
    return K.utils.to_categorical(labels, num_classes=classes)
