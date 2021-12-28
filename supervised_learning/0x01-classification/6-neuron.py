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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        """
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron:
        derivative of loss function with respect to A:
            dA = (-Y / A) + ((1 - Y) / (1 - A))
        derivative of A with respect to z:
            dz = A * (1 - A)
        combining two above with chain rule,
        derivative of loss function with respect to z:
            dz = A - Y
        using chain rule with above derivative,
        derivative of loss function with respect to __W:
            dW = Xdz
        derivative of loss function with respect to __b:
            db = dz
        one-step of gradient descent updates the attributes with the following:
            __W = __W - (alpha * dW)
            __b = __b - (alpha * db)
        """
        dz = A - Y
        dw = np.matmul(X, dz.T) / A.shape[1]
        db = np.sum(dz) / A.shape[1]
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)

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
