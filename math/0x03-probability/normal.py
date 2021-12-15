#!/usr/bin/env python3
"""Create a class Normal that represents a normal distribution"""


class Normal:
    """class that represents normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """class constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = sum(map(lambda i: (i - self.mean) ** 2, data))
            self.stddev = (self.stddev / len(data)) ** (1 / 2)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Probability Density Function"""
        e = 2.7182818285
        pi = 3.1415926536
        exponent = -0.5 * (((x - self.mean) / self.stddev) ** 2)
        coeficient = 1 / (self.stddev * (2 * pi) ** (1 / 2))
        pdf = coeficient * e ** exponent
        return pdf
