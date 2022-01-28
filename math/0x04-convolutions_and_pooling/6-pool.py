#!/usr/bin/env python3
"""
Function that performs pooling on images
"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
      m is the number of images
      h is the height in pixels of the images
      w is the width in pixels of the images
      c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing
    the kernel shape for the pooling
      kh is the height of the kernel
      kw is the width of the kernel
    stride is a tuple of (sh, sw)
      sh is the stride for the height of the image
      sw is the stride for the width of the image
    mode indicates the type of pooling
      max indicates max pooling
      avg indicates average pooling
    Returns: a numpy.ndarray containing the pooled images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh, kw = kernel_shape
    sh, sw = stride
    o_h = (h - kh) // sh + 1
    o_w = (w - kw) // sw + 1
    output = np.zeros((m, o_h, o_w, c))
    for x in range(o_w):
        for y in range(o_h):
            i = y * sh
            j = x * sw
            mat = images[:, i:i+kh, j:j+kw, :]
            if mode == 'max':
                output[:, y, x, :] = np.max(mat, axis=(1, 2))
            elif mode == 'avg':
                output[:, y, x, :] = np.average(mat, axis=(1, 2))
    return output
