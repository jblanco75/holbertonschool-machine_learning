#!/usr/bin/env python3
"""
Function that performs a valid convolution on grayscale images
"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
      m is the number of images
      h is the height in pixels of the images
      w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
      kh is the height of the kernel
      kw is the width of the kernel
    padding is a tuple of (ph, pw)
      ph is the padding for the height of the image
      pw is the padding for the width of the image
    the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph, pw = padding
    new_padded_images = np.pad(images, ((0, 0),
                                        (ph, ph),
                                        (pw, pw)))
    o_h = h + (2 * ph) - kh + 1
    o_w = w + (2 * pw) - kw + 1
    output = np.zeros((m, o_h, o_w))
    for x in range(o_w):
        for y in range(o_h):
            mat = new_padded_images[:, y:y+kh, x:x+kw]
            output[:, y, x] = np.sum(np.multiply(mat, kernel), axis=(1, 2))
    return output
