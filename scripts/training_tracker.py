import yaml
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(__file__)  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.append(parent_dir)

from neural_network import Layer, Network
from neural_network import Tanh, ReLU, Softmax, Linear, Sigmoid
from neural_network import MSE, MAE, L1, L2

if __name__ == '__main__':
    # Generate data
    X = np.linspace(-1, 1, 100)  # 100 data points between -1 and 1
    X = X.reshape(1, -1)  # Reshape to a column vector
    Y = 2 * X + 1  # Linear relationship with some noise

    # Define network
    layers = [Layer(1, 10, activation_function=Tanh()),
              Layer(10, 1, activation_function=Linear())]
    
    network = Network(layers, loss_function=MSE())

    predictions = []

    # Initial predictions
    predictions.append(network.predict(X))

    # Train network
    network.fit(X, Y, epochs=10000, learning_rate=0.01, verbose=0, batch_size=100)

    # Final predictions
    predictions.append(network.predict(X))
    losses = network.losses
    
    # Plot results
    # plt.plot(layers[0].weights_history, label='Layer 1')
    # plt.plot(layers[1].weights_history, label='Layer 2')
    # plt.plot(layers[0].gradients, label='Layer 1')
    # plt.plot(layers[1].gradients, label='Layer 2')
    # plt.legend()
    # plt.show()
    plt.plot(X.T, Y.T, label='True')
    plt.plot(X.T, predictions[0].T, label='Initial')
    plt.plot(X.T, predictions[-1].T, label='Final')
    plt.legend()
    plt.show()
    plt.plot(losses)
    plt.title('Loss over time')
    plt.show()
