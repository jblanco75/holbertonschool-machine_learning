#!/usr/bin/env python3
"""
Defines class MultiNormal
"""


import numpy as np


class MultiNormal:
    """
    Class that represents Multivariate Normal Distribution
    """
    def __init__(self, data):
        """
        Class constructor
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov
        self.stdev = np.std(data, axis=1)

    def pdf(self, x):
        """
        public instance method that calculates the PDF at a data point
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        pdf = (1.0 / (self.stdev *
                      np.sqrt(2*np.pi))) * np.exp(-0.5*((x - self.mean) /
                                                          self.stdev) ** 2)
        return pdf
