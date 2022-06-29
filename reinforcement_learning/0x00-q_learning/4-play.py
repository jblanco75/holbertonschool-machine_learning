#!/usr/bin/env python3
"""
Function that has trained agent play an episode
"""


import gym
import numpy as np


def play(env, Q, max_steps=100):
    """
    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    max_steps is the maximum number of steps in the episode
    Each state of the board should be displayed via the console
    You should always exploit the Q-table
    Returns: the total rewards for the episode
    """
    current_state = env.reset()
    done = False
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[current_state, :])
        next_state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
        current_state = next_state
    return reward
