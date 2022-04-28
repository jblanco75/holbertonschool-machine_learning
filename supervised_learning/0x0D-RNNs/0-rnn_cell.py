#!/usr/bin/env python3
"""
Creation of the class RNNCell
"""


import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        class Constructor
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        """
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(x_concat, self.Wh) + self.bh
        h_next = np.tanh(h_next)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
