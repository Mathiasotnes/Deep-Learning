import numpy as np
"""
This Python file defines a framework for neural network activation functions. 
It includes a base class Activation_function, serving as an abstract class with 
two methods calculate() and derivative(), both of which are intended to be overridden in derived classes.

Activation functions included are:
    - ReLU
    - Sigmoid
    - Tanh
    - Softmax
    - Linear

"""


class Activation_function():

    def calculate(self):
        raise NotImplementedError()

    def derivative(self):
        raise NotImplementedError()


class ReLU(Activation_function):

    def calculate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid(Activation_function):

    def calculate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.calculate(x) * (1 - self.calculate(x))


class Tanh(Activation_function):

    def calculate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2


class Softmax(Activation_function):
    
    def calculate(self, x):
        shiftx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, x):
        return self.calculate(x) * (1 - self.calculate(x))


class Linear(Activation_function):

    def calculate(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)
