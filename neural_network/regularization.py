import numpy as np
"""
This Python file defines a framework for neural network regularization functions. 
It includes a base class Regulator, serving as an abstract class with 
two methods penalty() and derivative(), both of which are intended to be overridden in derived classes.

Regularization functions included are:
    - L1
    - L2

"""


class Regulator():

    def __init__(self, _lambda):
        self._lambda = _lambda

    def penalty(self, weights):
        raise NotImplementedError()

    def derivative(self, weights):
        raise NotImplementedError()


class L1(Regulator):

    def penalty(self, weights):
        return self._lambda * np.sum(np.abs(weights))

    def derivative(self, weights):
        return self._lambda * np.sign(weights)


class L2(Regulator):

    def penalty(self, weights):
        return self._lambda * np.sum(weights**2)

    def derivative(self, weights):
        return 2 * self._lambda * weights
