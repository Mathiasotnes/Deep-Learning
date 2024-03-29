import time
import numpy as np
from utilities import calculate_time
import matplotlib.pyplot as plt


class Layer:

    def __init__(self, input_dim, output_dim, activation_function, weight_std=1):
        self.weights = np.random.randn(output_dim, input_dim) * weight_std
        self.biases = np.zeros((output_dim, 1))
        self.activation_function = activation_function
        self.Z = None
        self.X = None
        self.gradients = np.ndarray([])
        self.weights_history = np.ndarray([])

    def forward_pass(self, X):
        self.X = X  # Save the input for backpropagation
        self.Z = np.dot(self.weights, X.T).T + self.biases.T  # Z = W'x + b (sum of weightet inputs)
        return self.activation_function.calculate(self.Z)  # output = activation(Z) (output of layer)

    def backward_pass(self, dLoss_dOut):

        # The derivative is w.r.t. Z (sum of weightet inputs, delta)
        dOut_dZ = self.activation_function.derivative(self.Z)

        # Chain rule
        dLoss_dZ = dLoss_dOut * dOut_dZ

        # Gradient w.r.t. weights (X = dZ/dW)
        dLoss_dW = np.dot(self.X.T, dLoss_dZ)

        # Gradient w.r.t. inputs (W = dZ/dX)
        dLoss_dX = np.dot(dLoss_dZ, self.weights)

        # Gradient w.r.t. biases (b = dZ/db)
        dLoss_db = np.sum(dLoss_dZ, axis=0, keepdims=True)

        # Save gradients for visualization
        self.gradients = np.append(self.gradients, np.mean(dLoss_dZ))
        self.weights_history = np.append(self.weights_history,
                                         np.mean(self.weights))

        return dLoss_dX, dLoss_dW, dLoss_db

    def update_parameters(self,
                          dLoss_dW,
                          dLoss_db,
                          learning_rate,
                          regularization=None):

        # Add regularization
        if regularization:
            dLoss_dW += regularization.derivative(self.weights.T)
            dLoss_db += regularization.derivative(self.biases.T)

        # Update weights and biases
        self.weights -= learning_rate * dLoss_dW.T
        self.biases -= learning_rate * dLoss_db.T


class Network():

    def __init__(self, layers, loss_function, regularization=None):
        self.layers = layers
        self.loss_function = loss_function
        self.regularization = regularization
        self.train_loss = []
        self.val_loss = []

    def predict(self, X):
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
            X_val=None,
            y_val=None,
            learning_rate=0.01,
            epochs=10,
            verbose=0,
            batch_size=32):

        # Start timer
        start_time = time.time()
        last_update_time = None

        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            train_losses = []

            # Mini-batch training
            for i in range(0, X_train.shape[0], batch_size):
                # Create a mini-batch
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                # Forward and backward pass
                y_pred = self.predict(X_batch)
                loss = self.loss_function.calculate(y_batch, y_pred)
                dLoss_dOut = self.loss_function.derivative(y_batch, y_pred)
                self.__backprop(dLoss_dOut, learning_rate)
                train_losses.append(loss)

                if verbose == 1:
                    last_update_time = self.print_progress_bar(
                        epoch + 1,
                        epochs,
                        loss=loss,
                        length=50,
                        last_update_time=last_update_time,
                        final=(epoch == epochs - 1))

            last_loss = np.mean(train_losses)
            self.train_loss.append(last_loss)
            if (X_val is not None) and (y_val is not None):
                y_pred = self.predict(X_val)
                val_loss = self.loss_function.calculate(y_val, y_pred)
                self.val_loss.append(val_loss)
                if verbose == 2:
                    training_time = time.time() - start_time
                    np.set_printoptions(precision=3, suppress=True)
                    first_val = y_val[0].astype(int)
                    print(f"Val Loss: {val_loss:.6f}  | Train Loss: {last_loss:.6f}  |  Outputs: {y_pred[0]}  |  Targets: {first_val}  |  Time: {training_time:.2f} seconds")


        if verbose >= 1:
            training_time = time.time() - start_time
            print(f"\nTraining Time: {training_time:.2f} seconds")



"""
Things that must be printet out on verbose level 2:
network inputs, network outputs, target values, and error/loss

Things to prepare:

A configuration file for a network with at least 2 hidden layers that runs on a dataset of 
at least 500 training cases and that has previously shown some learning progress 
(i.e. loss clearly declines over time / minibatches). 
All parameters of this network should be tuned to those that have worked well in the past.

A configuration file for a network with no hidden layers that runs on the same 500-item 
training set as above. This network may or may not exhibit learning progress.

A configuration file for a network with at least 5 hidden layers that runs on a dataset 
of at least 100 training cases for a minimum of 10 passes through the entire training set.
"""
