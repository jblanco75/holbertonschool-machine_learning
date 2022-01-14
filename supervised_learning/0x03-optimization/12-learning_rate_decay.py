#!/usr/bin/env python3
"""
Function that creates a learning rate decay
operation in tensorflow using inverse time decay
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Returns: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
