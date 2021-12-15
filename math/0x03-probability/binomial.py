#!/usr/bin/env python3
"""Create a class Binomial that represents a binomial distribution"""


class Binomial:
    """class that represents Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """class constructor"""
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            else:
                self.n = int(n)
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum(map(lambda i: (i - mean) ** 2, data)) / len(data)
            self.p = 1 - ((var) / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """Probability Mass Function"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        q = (1 - self.p)
        factorial_n = 1
        for i in range(self.n):
            factorial_n *= (i + 1)
        factorial_k = 1
        for i in range(k):
            factorial_k *= (i + 1)
        factorial_n_k = 1
        for i in range(self.n - k):
            factorial_n_k *= (i + 1)
        bin_co = factorial_n / (factorial_k * factorial_n_k)
        pmf = bin_co * (self.p ** k) * (q ** (self.n - k))
        return pmf

    def cdf(self, k):
        """Cumulative Distribution Function"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
