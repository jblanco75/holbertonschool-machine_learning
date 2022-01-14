#!/usr/bin/env python3
"""
Function that creates a learning rate decay
operation in tensorflow using inverse time decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Returns: the learning rate decay operation
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
