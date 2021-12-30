#!/usr/bin/env python3
"""
Class that defines a deep neural network
performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines the DeepNeuralNetwork class
    """
    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')
        self.L = len(layers)
        self.cache = {}
        weights = {}
        prev = nx
        for index, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, prev) * np.sqrt(2 / prev))
            prev = layer
        self.weights = weights
