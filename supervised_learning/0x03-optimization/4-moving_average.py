#!/usr/bin/env python3
"""
Function that calculates the weighted moving average of a data set
"""


import numpy as np


def moving_average(data, beta):
    """
    Returns: a list containing the moving averages of data
    """
    vt = 0
    weighted_list = []
    for i in range(len(data)):
        vt = (vt * beta + (1 - beta) * data[i])
        weighted_list.append(vt / (1 - beta**(i + 1)))
    return weighted_list
