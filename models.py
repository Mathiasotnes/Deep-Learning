import numpy as np
import losses
import activations
from utilities import calculate_time


class Layer:

    def __init__(self,
                 input_dim,
                 output_dim,
                 activation_function=activations.ReLU()):
        self.weights = np.random.randn(output_dim, input_dim)
        self.biases = np.zeros((output_dim,1))
        self.activation_function = activation_function
        self.Z = None

    def forward_pass(self, X):
        self.Z = np.dot(self.weights, X) + self.biases  # Z = Wx + b
        return self.activation_function.calculate(self.Z)

    def backward_pass(self, dLoss_dOut, X):

        # The derivative is w.r.t. Z
        dOut_dZ = self.activation_function.derivative(self.Z)  

        # Chain rule
        dLoss_dZ =  dLoss_dOut * dOut_dZ

        # Gradient w.r.t. weights
        dLoss_dW = X * dLoss_dZ

        # Gradient w.r.t. inputs
        dLoss_dX = np.dot(dLoss_dZ.T, self.weights).T

        # Gradient w.r.t. biases
        dLoss_db = np.sum(dLoss_dZ, axis=0, keepdims=True)

        return dLoss_dX, dLoss_dW, dLoss_db

    def update_parameters(self, dLoss_dW, dLoss_db, learning_rate):
        # Average the gradients over the batch
        dLoss_dW_mean = np.mean(dLoss_dW, axis=1, keepdims=True)
        dLoss_db_mean = np.mean(dLoss_db, axis=1, keepdims=True)

        # Update weights and biases
        self.weights -= learning_rate * dLoss_dW_mean
        self.biases -= learning_rate * dLoss_db_mean


class Network():

    def __init__(self, layers, learning_rate=1, loss_function=losses.MSE()):
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def __backprop(self, dLoss_dOut, learning_rate):
        for layer in reversed(self.layers):
            dLoss_dOut, dLoss_dW, dLoss_db = layer.backward_pass(
                dLoss_dOut, layer.Z)
            layer.update_parameters(dLoss_dW, dLoss_db, learning_rate)

    @calculate_time
    def fit(self, X_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.predict(X_train)
            loss = self.loss_function.calculate(y_train, y_pred)
            dLoss_dOut = self.loss_function.derivative(y_train, y_pred)
            self.__backprop(dLoss_dOut, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
