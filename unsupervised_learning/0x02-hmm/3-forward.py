#!/usr/bin/env python3
"""
Function that performs the forward algorithm for a hidden markov model
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Observation is a numpy.ndarray of shape (T,) that contains the index
    of the observation
      T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
      Emission[i, j] is the probability of observing j given the hidden state i
      N is the number of hidden states
      M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
      Transition[i, j] is the probability of transitioning from the
      hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state
    Returns: P, F, or None, None on failure
      P is the likelihood of the observations given the model
      F is a numpy.ndarray of shape (N, T) containing the forward path
      probabilities
        F[i, j] is the probability of being in hidden state i at time j
        given the previous observations
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
    F = np.zeros((N, T))
    index = Observation[0]
    Emission_idx = Emission[:, index]
    F[:, 0] = Initial.T * Emission_idx
    for j in range(1, T):
        for i in range(N):
            F[i, j] = np.sum(Emission[i, Observation[j]]
                             * Transition[:, i] * F[:, j - 1], axis=0)
    P = np.sum(F[:, T-1:], axis=0)[0]
    return P, F
