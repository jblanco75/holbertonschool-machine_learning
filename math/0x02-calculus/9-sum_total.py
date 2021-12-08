#!/usr/bin/env python3
"""Function sigma for a squared positive int"""


def summation_i_squared(n):
    """Returns sum of a squared positive int"""
    result = 0
    if type(n) is not int or n < 1:
        return None
    else:
        nums = range(n+1)
        result = map(lambda n: n**2, nums)
    return sum(result)
