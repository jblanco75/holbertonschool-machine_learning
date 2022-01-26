#!/usr/bin/env python3
"""
Functions to save and load weights
"""


import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves model config in a JSON file
    """
    model_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(model_config)
    return None


def load_config(filename):
    """
    Loads model config from JSON file
    """
    with open(filename, 'r') as f:
        model_config = f.read()
    return K.models.model_from_json(model_config)
