#!/usr/bin/env python3
"""Create a class Exponential that represents an exponential distribution"""


class Exponential:
    """Class that represents exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                lambtha = float(len(data) / sum(data))
                self.lambtha = lambtha

    def pdf(self, x):
        """Probability Density Function"""
        if x < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        pdf = lambtha * (e ** (-lambtha * x))
        return pdf

    def cdf(self, x):
        """Cumulative Distribution Function"""
        if x < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        cdf = 1 - (e ** (-lambtha * x))
        return cdf
