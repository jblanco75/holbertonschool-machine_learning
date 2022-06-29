#!/usr/bin/env python3
"""
Function that initializes the Q-table with environment instance
"""


import gym
import numpy as np


def q_init(env):
    """
    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    Q_table = np.zeros((env.observation_space.n, env.action_space.n))
    return Q_table
