#!/usr/bin/env python3
"""
Function that creates the training operation for
a neural network in tensorflow using the RMSProp
optimization algorithm
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Returns: the RMSProp optimization operation
    """
    rms_optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                              decay=beta2, epsilon=epsilon)
    trainer = rms_optimizer.minimize(loss)
    return trainer
