#!/usr/bin/env python3
"""
Function that creates a batch normalization
layer for a neural network in tensorflow
"""


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_batch_norm_layer(prev, n, activation):
    """
    Returns: a tensor of the activated output for the layer
    """
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.Dense(n, kernel_initializer=k)
    mean, var = tf.nn.moments(base(prev), axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(base(prev), mean, var,
                                           beta, gamma, epsilon)
    return activation(batch_norm)
