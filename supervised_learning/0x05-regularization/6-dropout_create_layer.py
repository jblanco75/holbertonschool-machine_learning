#!/usr/bin/env python3
"""
Function that creates a layer of a neural network using dropout
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Returns: the output of the new layer
    """
    weights = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=weights, name='layer')
    dropout = tf.layers.Dropout(keep_prob)
    return dropout(layer(prev))
