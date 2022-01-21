#!/usr/bin/env python3
"""
Function that determines if you should stop gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Returns: a boolean of whether the network should be
    stopped early, followed by the updated count
    """
    if opt_cost - cost > threshold:
        return False, 0
    else:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count
