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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        y = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-y))
        return (self.__A)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost_sum = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = - np.sum(cost_sum) / m
        return cost

    @property
    def W(self):
        """
        Getter
        __W The weights vector for the neuron
        """
        return (self.__W)

    @property
    def b(self):
        """
        Getter
        __b bias for the neuron
        """
        return (self.__b)

    @property
    def A(self):
        """
        Getter
        __A The activated output of the neuron
        """
        return (self.__A)
