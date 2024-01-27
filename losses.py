import numpy as np


class loss_function():
    """
    A base class for loss functions acting as an
    abstract class. 
    """

    def calculate(self, y_true, y_pred):
        raise NotImplementedError()

    def derivative(self, y_true, y_pred):
        raise NotImplementedError()


class MSE(loss_function):

    def calculate(self, y_true, y_pred):
        return np.mean(.5 * (y_true - y_pred)**2)

    def derivative(self, y_true, y_pred):
        return y_pred - y_true
