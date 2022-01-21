#!/usr/bin/env python3
"""
Function that calculates the cost
of a neural network with L2 regularization
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def l2_reg_cost(cost):
    """
    Returns: a tensor containing the cost of
    the network accounting for L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
