#!/usr/bin/env python3
"""
Function that calculates the softmax cross-entropy loss of a prediction
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """Returns: a tensor containing the loss of the prediction"""
    loss = tf.keras.losses.CategoricalCrossentropy(y, y_pred)
    return loss
