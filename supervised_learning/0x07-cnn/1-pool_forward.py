#!/usr/bin/env python3
"""
Function that performs forward propagation
over a pooling layer of a neural network
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
      m is the number of examples
      h_prev is the height of the previous layer
      w_prev is the width of the previous layer
      c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of
    the kernel for the pooling
      kh is the kernel height
      kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
      sh is the stride for the height
      sw is the stride for the width
    mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride[0], stride[1]

    o_h = ((h_prev - kh) // sh) + 1
    o_w = ((w_prev - kw) // sw) + 1

    output_dim = (m, o_h, o_w, c_prev)
    convolution = np.zeros(output_dim)

    for x in range(o_w):
        for y in range(o_h):
            i = y * sh
            j = x * sw
            mat = A_prev[:, i:i+kh, j:j+kw, :]
            if mode == 'max':
                convolution[:, y, x, :] = np.max(mat, axis=(1, 2))
            else:
                convolution[:, y, x, :] = np.mean(mat, axis=(1, 2))
    return convolution
