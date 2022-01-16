#!/usr/bin/env python3
"""
Function that creates a batch normalization
layer for a neural network in tensorflow
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_batch_norm_layer(prev, n, activation):
    """
    Returns: a tensor of the activated output for the layer
    """
    k_init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, kernel_initializer=k_init)
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True,
                        name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True,
                       name='beta')
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(layer(prev), mean, variance,
                                           beta, gamma, epsilon)
    return activation(batch_norm)
