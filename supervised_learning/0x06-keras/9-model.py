#!/usr/bin/env python3
"""
Functions to save and load a model
"""


import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves the model
    Returns None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads the model
    Returns the model
    """
    return K.models.load_model(filename)
