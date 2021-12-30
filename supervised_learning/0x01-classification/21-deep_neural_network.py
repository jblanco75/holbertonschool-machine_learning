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
        inputs = nx
        self.__L = len(layers)
        self.__cache = {}
        weights = {}
        for index, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, inputs) * np.sqrt(2 / inputs))
            inputs = layer
        self.__weights = weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache["A0"] = X
        for index in range(self.L):
            W = self.weights["W{}".format(index + 1)]
            b = self.weights["b{}".format(index + 1)]
            z = np.matmul(W, self.cache["A{}".format(index)]) + b
            A = 1 / (1 + (np.exp(-z)))
            self.__cache["A{}".format(index + 1)] = A
        return (A, self.cache)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        A, cache = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        retro = {}
        for idx in range(self.L, 0, -1):
            A = cache["A{}".format(idx - 1)]
            if idx == self.L:
                retro["dz{}".format(idx)] = (cache["A{}".format(idx)] - Y)
            else:
                dz_1 = retro["dz{}".format(idx + 1)]
                A_1 = cache["A{}".format(idx)]
                retro["dz{}".format(idx)] = (
                    np.matmul(W_1.transpose(), dz_1) *
                    (A_1 * (1 - A_1)))
            dz = retro["dz{}".format(idx)]
            dW = np.matmul(dz, A.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            W_1 = self.weights["W{}".format(idx)]
            self.__weights["W{}".format(idx)] = (
                self.weights["W{}".format(idx)] - (alpha * dW))
            self.__weights["b{}".format(idx)] = (
                self.weights["b{}".format(idx)] - (alpha * db))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
