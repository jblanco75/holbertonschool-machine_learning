#!/usr/bin/env python3
"""
Creates the class SelfAttention
"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Class to encode for machine translation:
    """

    def __init__(self, units):
        """
        Class constructor
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder
        Returns: context, weights
            context is a tensor of shape (batch, units) that contains
            the context vector for the decoder
            weights is a tensor of shape (batch, input_seq_len, 1) that
            contains the attention weights
        """
        s_expanded = tf.expand_dims(input=s_prev, axis=1)
        inputs = self.U(s_expanded)
        hidden = self.W(hidden_states)
        score = self.V(tf.nn.tanh(inputs + hidden))

        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * s_expanded
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights
