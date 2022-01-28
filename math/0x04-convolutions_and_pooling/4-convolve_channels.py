#!/usr/bin/env python3
"""
Function that performs a convolution on
images with channels
"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
      m is the number of images
      h is the height in pixels of the images
      w is the width in pixels of the images
      c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c) containing the kernel
    for the convolution
      kh is the height of the kernel
      kw is the width of the kernel
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
    kh = kernel.shape[0]
    kw = kernel.shape[1]
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
    o_h = ((h + (2 * ph) - kh) // sh) + 1
    o_w = ((w + (2 * pw) - kw) // sw) + 1
    output = np.zeros((m, o_h, o_w))
    for x in range(o_w):
        for y in range(o_h):
            i = y * sh
            j = x * sw
            mat = new_padded_images[:, i:i+kh, j:j+kw, :]
            output[:, y, x] = np.sum(np.multiply(mat, kernel), axis=(1, 2, 3))
    return output
