#!/usr/bin/env python3
"""
Function that creates the training operation
for a neural network in tensorflow using the
Adam optimization algorithm
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Returns: the Adam optimization operation
    """
    adam_optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    trainer = adam_optimizer.minimize(loss)
    return trainer
