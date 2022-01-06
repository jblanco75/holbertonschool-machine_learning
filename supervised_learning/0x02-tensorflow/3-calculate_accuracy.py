#!/usr/bin/env python3
"""
Function that calculates the accuracy of a prediction
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
