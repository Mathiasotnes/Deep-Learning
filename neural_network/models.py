import time
import numpy as np
from utilities import calculate_time


class Layer:

    def __init__(self, input_dim, output_dim, activation_function):
        self.weights = np.random.randn(output_dim, input_dim)
        self.biases = np.zeros((output_dim, 1))
        self.activation_function = activation_function
        self.Z = None
        self.X = None

    def forward_pass(self, X):
        self.X = X  # Save the input for backpropagation
        self.Z = np.dot(self.weights, X) + self.biases  # Z_ = W'x + b
        return self.activation_function.calculate(self.Z)  # Z = af(Z_)

    def backward_pass(self, dLoss_dOut):

        # The derivative is w.r.t. Z
        dOut_dZ = self.activation_function.derivative(self.Z)

        # Chain rule
        dLoss_dZ = dLoss_dOut * dOut_dZ

        # Gradient w.r.t. weights, averaged over the batch
        dLoss_dW = np.dot(dLoss_dZ, self.X.T)
        dLoss_dW_mean = dLoss_dW / self.X.shape[1]

        # Gradient w.r.t. inputs
        dLoss_dX = np.dot(dLoss_dZ.T, self.weights).T

        # Gradient w.r.t. biases, averaged over the batch
        dLoss_db = np.sum(dLoss_dZ, axis=0, keepdims=True)
        dLoss_db_mean = np.mean(dLoss_db, axis=1, keepdims=True)

        return dLoss_dX, dLoss_dW_mean, dLoss_db_mean

    def update_parameters(self,
                          dLoss_dW,
                          dLoss_db,
                          learning_rate,
                          regularization=None):

        # Add regularization
        if regularization:
            dLoss_dW += regularization.derivative(self.weights)

        # Update weights and biases
        self.weights -= learning_rate * dLoss_dW
        self.biases -= learning_rate * dLoss_db


class Network():

    def __init__(self, layers, loss_function, regularization=None):
        self.layers = layers
        self.loss_function = loss_function
        self.regularization = regularization

    def predict(self, X):
        # Check if the input shape matches the expected shape of the network's first layer
        if len(X.shape) == 2 and self.layers[0].weights.shape[1] != X.shape[0]:
            # Transpose X if the number of features (second dimension of weights) does not match the first dimension of X
            X = X.T

        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def __backprop(self, dLoss_dOut, learning_rate):
        for layer in reversed(self.layers):
            dLoss_dOut, dLoss_dW, dLoss_db = layer.backward_pass(dLoss_dOut)
            layer.update_parameters(dLoss_dW, dLoss_db, learning_rate,
                                    self.regularization)

    def print_progress_bar(self,
                           iteration,
                           total,
                           loss=None,
                           length=50,
                           fill='█',
                           last_update_time=None,
                           final=False):
        current_time = time.time()
        if final or last_update_time is None or (current_time -
                                                 last_update_time >= 1):
            percent = ("{0:.1f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + '-' * (length - filled_length)
            print(f'\r|{bar}| {percent}% Complete  |  Loss: {loss:.6f}',
                  end='\r')
            return current_time  # Return the time of this update
        return last_update_time  # Return the previous update time if no update was made

    def fit(self,
            X_train,
            y_train,
            learning_rate,
            epochs,
            verbose=0,
            batch_size=32):

        # Start timer
        start_time = time.time()
        last_update_time = None

        # Ensure y_train has the correct shape (1, samples)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(1, -1)

        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation, :]
            y_train_shuffled = y_train[:, permutation]

            # Mini-batch training
            for i in range(0, X_train.shape[0], batch_size):
                # Create a mini-batch
                X_batch = X_train_shuffled[
                    i:i + batch_size, :].T  # Transpose to match expected shape
                y_batch = y_train_shuffled[:, i:i + batch_size]

                # Forward and backward pass
                y_pred = self.predict(
                    X_batch
                )  # Assumes predict function expects shape (features, samples)
                loss = self.loss_function.calculate(y_batch, y_pred)
                dLoss_dOut = self.loss_function.derivative(y_batch, y_pred)
                self.__backprop(dLoss_dOut, learning_rate)

                if verbose >= 1:
                    last_update_time = self.print_progress_bar(
                        epoch + 1,
                        epochs,
                        loss=loss,
                        length=50,
                        last_update_time=last_update_time,
                        final=(epoch == epochs - 1))

        training_time = time.time() - start_time
        print(f"\nTraining Time: {training_time:.2f} seconds")
