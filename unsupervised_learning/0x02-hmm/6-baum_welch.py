#!/usr/bin/env python3
"""
Function that performs the Baum-Welch algorithm for a hidden markov model
"""


import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Observations is a numpy.ndarray of shape (T,) that contains the index of
    the observation
      T is the number of observations
    Transition is a numpy.ndarray of shape (M, M) that contains the
    initialized transition probabilities
      M is the number of hidden states
    Emission is a numpy.ndarray of shape (M, N) that contains the
    initialized emission probabilities
      N is the number of output states
    Initial is a numpy.ndarray of shape (M, 1) that contains the initialized
    starting probabilities
    iterations is the number of times expectation-maximization should be
    performed
    Returns: the converged Transition, Emission, or None, None on failure
    """
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    if not np.isclose(np.sum(Initial, axis=0), [1])[0]:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None
    if not np.isclose(np.sum(Transition, axis=1),
                      np.ones(Initial.shape[0])).all():
        return None, None
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None
    if not isinstance(iterations, int) or iterations < 0:
        return None, None

    N = Initial.shape[0]
    T = Observations.shape[0]
    M = Emission.shape[1]

    a = Transition
    b = Emission
    a_prev = np.copy(a)
    b_prev = np.copy(b)

    for iteration in range(1000):
        PF, F = forward(Observations, b, a, Initial)
        PB, B = backward(Observations, b, a, Initial)
        X = np.zeros((N, N, T - 1))
        NUM = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    Fit = F[i, t]
                    aij = a[i, j]
                    bjt1 = b[j, Observations[t + 1]]
                    Bjt1 = B[j, t + 1]
                    NUM[i, j, t] = Fit * aij * bjt1 * Bjt1
        DEN = np.sum(NUM, axis=(0, 1))
        X = NUM / DEN
        G = np.zeros((N, T))
        NUM = np.zeros((N, T))
        for t in range(T):
            for i in range(N):
                Fit = F[i, t]
                Bit = B[i, t]
                NUM[i, t] = Fit * Bit
        DEN = np.sum(NUM, axis=0)
        G = NUM / DEN

        a = np.sum(X, axis=2) / np.sum(G[:, :T - 1], axis=1)[..., np.newaxis]
        DEN = np.sum(G, axis=1)
        NUM = np.zeros((N, M))
        for k in range(M):
            NUM[:, k] = np.sum(G[:, Observations == k], axis=1)
        b = NUM / DEN[..., np.newaxis]

        if np.all(np.isclose(a, a_prev)) or np.all(np.isclose(a, a_prev)):
            return a, b
        a_prev = np.copy(a)
        b_prev = np.copy(b)

    return a, b
