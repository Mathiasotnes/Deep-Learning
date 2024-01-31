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
"""
Usage Guide for function_approximator.py

This script trains a neural network based on a specified YAML configuration file. 
The configuration file should define the network architecture and training parameters.

Basic Usage:
    python3 function_approximator.py <path/to/config.yaml>

Command-line Arguments:
    - config: Mandatory. Path to the YAML configuration file.

Additional Flags (Optional):
    --verbose <level>: Sets the verbosity level of the output. 
        0: No verbose output (silent mode).
        1: Moderate verbosity (training progress).
        2: High verbosity (detailed training output).
    --save <path>: Saves the trained model to the specified path. Replace <path> with your desired file path.

Examples:
    Train a new model with no verbosity:
    python3 function_approximator.py <path/to/config.yaml> --verbose 0

    Train and save a new model with high verbosity:
    python3 function_approximator.py <path/to/config.yaml> --verbose 2 --save <path/to/save/model>
    python3 scripts/function_approximator.py scripts/configs/function_approximator.yml
"""

activation_functions = {
    'tanh': Tanh(),
    'relu': ReLU(),
    'softmax': Softmax(),
    'linear': Linear(),
    'sigmoid': Sigmoid()
}

loss_functions = {'mse': MSE(), 'mae': MAE()}

regularization_functions = {
    'l1': lambda rate: L1(rate),
    'l2': lambda rate: L2(rate)
}


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    # Define Layers
    layers = []
    for layer in config['network']['layers']:
        activation = activation_functions[layer['activation']]
        layers.append(
            Layer(layer['input'],
                  layer['output'],
                  activation_function=activation))

    # Define regularization
    regularization_type = regularization_functions[config['network']
                                                   ['regularization']['type']]
    regularization_rate = config['network']['regularization']['rate']

    # Define loss function
    loss_function = loss_functions[config['network']['loss']]

    # Define Network
    network = Network(layers,
                      loss_function=loss_function,
                      regularization=regularization_type(regularization_rate))

    return network

def generate_sine_data(samples=1000, amplitude=1, frequency=1, phase=0, noise=0.0):
    x = np.linspace(0, 2 * np.pi * frequency, samples).reshape(1, -1)
    y = amplitude * np.sin(x + phase) + noise * np.random.randn(samples)
    return x, y


def main(config, verbose=1, save_path=None, visualize=False):

    # Build model
    model = build_model(config)

    # Generate data
    x, y = generate_sine_data(samples=1000, amplitude=1, frequency=1, phase=0, noise=0.1)

    # Train model
    model.fit(x,
              y,
              learning_rate=config['training']['learning_rate'],
              epochs=config['training']['epochs'],
              verbose=verbose,
              batch_size=config['training']['batch_size'])

    # Evaluate model
    y_pred = model.predict(x)
    loss = model.loss_function.calculate(y, y_pred)
    print('Loss:', loss)
    #print('Accuracy:', np.mean(np.argmax(y_pred, axis=0) == y))

    # Save model
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    # Visualize results
    if visualize:
        y_pred = model.predict(x)  # Make sure x is in the correct shape
        plt.plot(x.flatten(), y.flatten(), label='True')
        plt.plot(x.flatten(), y_pred.flatten(), label='Predicted', linestyle='--')
        plt.legend()
        plt.title('Sine Function Approximation')
        plt.xlabel('x')
        plt.ylabel('sin(x)')
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a neural network based on YAML configuration.')

    # Add command-line arguments
    parser.add_argument('config',
                        type=str,
                        help='Path to the YAML configuration file.')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='Verbosity level of the output.')
    parser.add_argument('--save',
                        type=str,
                        default=None,
                        help='Path to save the trained model.')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='Visualize the results.')

    # Parse command-line arguments
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except:
        print('Error: Invalid configuration file.')
        exit()

    main(config, args.verbose, args.save, args.visualize)
