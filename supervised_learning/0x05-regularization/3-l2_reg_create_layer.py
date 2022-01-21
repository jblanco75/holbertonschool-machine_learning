#!/usr/bin/env python3
"""
Function that creates a tensorflow layer that includes L2 regularization
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Returns: the output of the new layer"""
    weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    l2_reg = tf.keras.regularizers.L2(l2=lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=weights,
                            kernel_regularizer=l2_reg, name='layer')
    return layer(prev)
