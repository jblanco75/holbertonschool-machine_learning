#!/usr/bin/env python3
"""
Function that calculates the most likely sequence of
hidden states for a hidden markov model
"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Observation is a numpy.ndarray of shape (T,) that contains the index of
    the observation
      T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
      Emission[i, j] is the probability of observing j given the hidden state i
      N is the number of hidden states
      M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities
      Transition[i, j] is the probability of transitioning from the hidden
      state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state
    Returns: path, P, or None, None on failure
      path is the a list of length T containing the most likely sequence of
      hidden states
      P is the probability of obtaining the path sequence
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
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None

    N = Initial.shape[0]
    T = Observation.shape[0]
    V = np.zeros((N, T))
    V[:, 0] = Initial.T * Emission[:, Observation[0]]
    B = np.zeros((N, T))

    for j in range(1, T):
        for i in range(N):
            temp = Emission[i, Observation[j]] * Transition[:, i] * V[:, j - 1]
            V[i, j] = np.max(temp, axis=0)
            B[i, j] = np.argmax(temp, axis=0)
    P = np.max(V[:, T - 1])
    S = np.argmax(V[:, T - 1])
    path = [S]
    for j in range(T - 1, 0, -1):
        S = int(B[S, j])
        path.append(S)
    path = path[:: -1]

    return path, P
