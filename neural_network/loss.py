import numpy as np
"""
This Python file defines a framework for neural network loss functions. 
It includes a base class Loss_function, serving as an abstract class with 
two methods calculate() and derivative(), both of which are intended to be overridden in derived classes.

Loss functions included are:
    - MSE
    - MAE

"""


class Loss_function():

    def calculate(self, y_true, y_pred):
        raise NotImplementedError()

    def derivative(self, y_true, y_pred):
        raise NotImplementedError()


class MSE(Loss_function):

    def calculate(self, y_true, y_pred):
        return np.mean(.5 * (y_true - y_pred)**2)

    def derivative(self, y_true, y_pred):
        return y_pred - y_true


class MAE(Loss_function):

    def calculate(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return np.where(y_pred > y_true, 1, -1)
    

class CrossEntropy(Loss_function):
    
    def calculate(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def derivative(self, y_true, y_pred):
        m = y_true.shape[0]
        dloss = y_pred - y_true
        dloss /= m
        return dloss
