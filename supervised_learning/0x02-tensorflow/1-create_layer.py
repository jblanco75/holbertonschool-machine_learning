#!/usr/bin/env python3
"""
Function to create a layer
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""
    weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=weights, name='layer')
    return layer(prev)
