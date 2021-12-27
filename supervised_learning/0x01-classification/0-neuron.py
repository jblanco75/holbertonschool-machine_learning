#!/usr/bin/env python3
"""
Class Neuron that defines a single neuron
performing binary classification
"""

import numpy as np


class Neuron:
    """class that defines a single neuron"""
    def __init__(self, nx):
        """Class Constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
