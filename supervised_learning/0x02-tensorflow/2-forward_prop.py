#!/usr/bin/env python3
"""
Function that creates the forward propagation graph for the neural network
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Returns: the prediction of the network in tensor form
    """
    for i in range(len(layer_sizes)):
        output = create_layer(x, layer_sizes[i], activations[i])
    return output
