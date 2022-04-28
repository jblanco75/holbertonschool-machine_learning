#!/usr/bin/env python3
"""
Function that performs forward propagation for a Deep RNN
"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    rnn_cells is a list of RNNCell instances of length l that will be
    used for the forward propagation
    l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
      t is the maximum number of time steps
      m is the batch size
      i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (l, m, h)
      h is the dimensionality of the hidden state
    Returns: H, Y
      H is a numpy.ndarray containing all of the hidden states
      Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, m, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        h_aux = X[step]
        for layer in range(len(rnn_cells)):
            r_cell = rnn_cells[layer]
            x_t = h_aux
            h_prev = H[step][layer]
            h_next, y_next = r_cell.forward(h_prev=h_prev, x_t=x_t)
            h_aux = h_next
            H[step + 1][layer] = h_aux
        Y[step] = y_next

    return H, Y
