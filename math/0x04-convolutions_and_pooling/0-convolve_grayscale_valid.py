#!/usr/bin/env python3
"""
Function that performs a valid convolution on grayscale images
"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w)
        containing multiple grayscale images
    m: is the number of images
    h: is the height in pixels of the images
    w: is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
        containing the kernel for the convolution
    kh: is the height of the kernel
    kw: is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    o_h = h - kh + 1
    o_w = w - kw + 1
    output = np.zeros((m, o_h, o_w))
    for x in range(o_w):
        for y in range(o_h):
            mat = images[:, y:y+kh, x:x+kw]
            output[:, y, x] = np.sum(np.multiply(mat, kernel), axis=(1, 2))
    return output
