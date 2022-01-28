#!/usr/bin/env python3
"""
Function that performs a convolution on images using multiple kernels
"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernels is a numpy.ndarray with shape (kh, kw, c, nc)
    containing the kernels for the convolution
    kh is the height of a kernel
    kw is the width of a kernel
    nc is the number of kernels
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]
    sh, sw = stride
    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    new_padded_images = np.pad(images, ((0, 0),
                                        (ph, ph),
                                        (pw, pw),
                                        (0, 0)), 'constant')
    o_h = ((h + 2 * ph - kh) // sh) + 1
    o_w = ((w + 2 * pw - kw) // sw) + 1
    output = np.zeros((m, o_h, o_w, nc))
    for x in range(o_w):
        for y in range(o_h):
            for z in range(nc):
                i = y * sh
                j = x * sw
                kernel = kernels[:, :, :, z]
                mat = new_padded_images[:, i:i+kh, j:j+kw, :]
                output[:, y, x, z] = np.sum(np.multiply(mat, kernel),
                                            axis=(1, 2, 3))
    return output
