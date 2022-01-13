#!/usr/bin/env python3
"""
Function that creates the training operation
for a neural network in tensorflow using the
gradient descent with momentum optimization algorithm
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_momentum_op(loss, alpha, beta1):
    """
    Returns: the momentum optimization operation
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
