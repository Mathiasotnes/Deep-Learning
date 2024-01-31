import yaml
import argparse
import numpy as np
import pickle
from neural_network import Layer, Network
from neural_network import Tanh, ReLU, Softmax, Linear, Sigmoid
from neural_network import MSE, MAE, L1, L2
from data_generation import Generator
"""
Usage Guide for image_classifier.py

This script trains a neural network based on a specified YAML configuration file. 
The configuration file should define the network architecture and training parameters.

Basic Usage:
    python3 image_classifier.py <path/to/config.yaml>

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
    python3 image_classifier.py <path/to/config.yaml> --verbose 0

    Train and save a new model with high verbosity:
    python3 image_classifier.py <path/to/config.yaml> --verbose 2 --save <path/to/save/model>
    python3 image_classifier.py config.yml
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


def generate_data(config):
    dataset_split = (config['data']['split']['train'],
                     config['data']['split']['val'],
                     config['data']['split']['test'])
    generator = Generator(image_size=config['data']['size'],
                          total_images=config['data']['quantity'],
                          dataset_split=dataset_split,
                          flatten=config['data']['flatten'])
    train_img, val_img, test_img = generator.generate(
        noise_level=config['data']['noise'])
    return train_img, val_img, test_img


def main(config, verbose=1, save_path=None):

    # Build model
    model = build_model(config)

    # Generate data
    train_img, val_img, test_img = generate_data(config)

    # Train model
    model.fit(train_img[0],
              train_img[1],
              learning_rate=config['training']['learning_rate'],
              epochs=config['training']['epochs'],
              verbose=verbose,
              batch_size=config['training']['batch_size'])

    # Evaluate model
    y_pred = model.predict(test_img[0])
    loss = model.loss_function.calculate(test_img[1], y_pred)
    print('Loss:', loss)
    print('Accuracy:', np.mean(np.argmax(y_pred, axis=0) == test_img[1]))

    # Save model
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)


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

    # Parse command-line arguments
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except:
        print('Error: Invalid configuration file.')
        exit()

    main(config, args.verbose, args.save)
