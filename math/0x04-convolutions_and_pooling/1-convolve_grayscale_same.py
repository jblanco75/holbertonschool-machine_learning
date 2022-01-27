#!/usr/bin/env python3
"""
Function that performs a same convolution on grayscale images
"""


import numpy as np


def convolve_grayscale_same(images, kernel):
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
    if necessary, the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    o_h = images.shape[1]
    o_w = images.shape[2]
    if kh % 2 != 0 and kw % 2 != 0:
        padding_h = (kh - 1) // 2
        padding_w = (kw - 1) // 2
    else:
        padding_h = kh // 2
        padding_w = kw // 2
    new_padded_images = np.pad(images, ((0, 0),
                                        (padding_h, padding_h),
                                        (padding_w, padding_w)))
    output = np.zeros((m, o_h, o_w))
    for x in range(o_w):
        for y in range(o_h):
            mat = new_padded_images[:, y:y+kh, x:x+kw]
            output[:, y, x] = np.sum(np.multiply(mat, kernel), axis=(1, 2))
    return output
