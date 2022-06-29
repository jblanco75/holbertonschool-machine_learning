#!/usr/bin/env python3
"""
Function that performs Q-learning
"""


import gym
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    When the agent falls in a hole, the reward should be updated to be -1
    Returns: Q, total_rewards
       Q is the updated Q-table
       total_rewards is a list containing the rewards per episode
    """
    total_rewards = []
    max_epsilon = epsilon
    for episode in range(episodes):
        current_state = env.reset()
        done = False

        total_episode_reward = 0

        for step in range(max_steps):
            p = np.random.uniform(0, 1)
            if p < epsilon:
                # exploring
                action = np.random.randint(Q.shape[1])
            else:
                # exploiting
                action = np.argmax(Q[current_state, :])

            next_state, reward, done, _ = env.step(action)

            if done and reward == 0:
                reward = -1

            Q[current_state, action] = (
                Q[current_state, action] * (1 - alpha) +
                alpha * (reward + gamma * np.max(Q[next_state, :])))
            total_episode_reward += reward

            if done:
                break

            current_state = next_state

        epsilon = (min_epsilon + (max_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))
        total_rewards.append(total_episode_reward)

    return Q, total_rewards
