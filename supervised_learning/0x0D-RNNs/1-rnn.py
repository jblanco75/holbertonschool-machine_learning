#!/usr/bin/env python3
"""
Function that performs forward propagation for a simple RNN
"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    rnn_cell is an instance of RNNCell that will be used for the forward
    propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
      t is the maximum number of time steps
      m is the batch size
      i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
      h is the dimensionality of the hidden state
    Returns: H, Y
      H is a numpy.ndarray containing all of the hidden states
      Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for i in range(t):
        x_t = X[i]
        h_prev = H[i]
        h_next, y_next = rnn_cell.forward(h_prev=h_prev, x_t=x_t)
        H[i + 1] = h_next
        Y[i] = y_next

    return H, Y
