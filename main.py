import matplotlib.pyplot as plt
import numpy as np
import utilities

"""
System must include the following objects:

Network object
Layer object
Data generator

System must include the following methods:
forward_pass
backward_pass

"""


class Activation_function():
    """
    A base class for activation functions acting as an
    abstract class. 
    """
    
    def calculate(self):
        raise NotImplementedError()
    
    def derivative(self):
        raise NotImplementedError()

class ReLU(Activation_function):
    def calculate(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Layer:
    def __init__(self, input_dim, output_dim, activation_function):
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.zeros((1, output_dim))
        self.activation_function = activation_function
        self.Z = None
    
    def forward_pass(self, X):
        self.Z = np.dot(self.weights, X) + self.biases  # Z = Wx + b
        return self.activation_function.calculate(self.Z)

    def backward_pass(self, dLoss_dOut, X):
        """
        Calculate the gradient of the loss with respect to weights and inputs.
        """
        dOut_dZ = self.activation_function.derivative(self.Z.T)  # The derivative is w.r.t. Z
        dLoss_dZ = dLoss_dOut * dOut_dZ  # Element-wise multiplication

        # Gradient w.r.t. weights
        dLoss_dW = np.dot(X, dLoss_dZ)
        
        # Gradient w.r.t. inputs
        dLoss_dX = np.dot(dLoss_dZ, self.weights.T)

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
    def __init__(self, layers, learning_rate=1):
        self.layers = layers
        self.learning_rate = learning_rate
    
    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X.T)
        return X
    
    def __backprop(self, dLoss_dOut, learning_rate):
        for layer in reversed(self.layers):
            dLoss_dOut, dLoss_dW, dLoss_db = layer.backward_pass(dLoss_dOut, layer.Z)
            layer.update_parameters(dLoss_dW, dLoss_db, learning_rate)
        
    def fit(self, X_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.predict(X_train).T
            loss = np.mean(.5 * (y_train - y_pred) ** 2)  # Mean squared error
            dLoss_dOut = y_pred - y_train
            self.__backprop(dLoss_dOut, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')


if __name__ == '__main__':
    # Generate dummy data
    X_train = np.linspace(-1, 1, 100).reshape(100, 1)
    y_train = 2 * X_train + 3 + np.random.randn(*X_train.shape) * 0.2

    # Create and train the network
    layer1 = Layer(1, 1, ReLU())
    network = Network([layer1])
    network.fit(X_train, y_train, learning_rate=0.001, epochs=1000)

    # Visualizing the results
    plt.figure(figsize=(10, 6))

    # Plot original data points
    plt.scatter(X_train, y_train, color='blue', label='Original data')

    # Generate predictions for plotting
    X_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_pred_plot = network.predict(X_plot)

    # Plot regression line
    plt.plot(X_plot, y_pred_plot.T, color='red', label='Fitted line')

    # Labeling the plot
    plt.xlabel('Input Feature')
    plt.ylabel('Target Value')
    plt.title('Linear Regression with Neural Network')
    plt.legend()

    # Show plot
    plt.show()
