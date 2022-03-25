#!/usr/bin/env python3
"""
Function that calculates the posterior prob of obtaining this
data given various hypothetical probabilities
"""


import numpy as np


def posterior(x, n, P, Pr):
    """
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
     probabilities of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P
    If n is not a positive integer, raise a ValueError with the message
     n must be a positive integer
    If x is not an integer that is greater than or equal to 0, raise a
     ValueError with the message x must be an integer that is greater
     than or equal to 0
    If x is greater than n, raise a ValueError with the message x cannot
     be greater than n
    If P is not a 1D numpy.ndarray, raise a TypeError with the message P
     must be a 1D numpy.ndarray
    If Pr is not a numpy.ndarray with the same shape as P, raise a TypeError
     with the message Pr must be a numpy.ndarray with the same shape as P
    If any value in P or Pr is not in the range [0, 1], raise a ValueError
     with the message All values in {P} must be in the range [0, 1] where
     {P} is the incorrect variable
    If Pr does not sum to 1, raise a ValueError with the message Pr must
     sum to 1
    All exceptions should be raised in the above order
    Returns: posterior probability of each probability in P,
     given x and n, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for value in range(P.shape[0]):
        if P[value] > 1 or P[value] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[value] > 1 or Pr[value] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")
    fact = np.math.factorial
    fact_coef = fact(n) / (fact(n - x) * fact(x))
    likelihood = fact_coef * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr
    marginal = np.sum(intersection)
    posterior = intersection / marginal
    return posterior
