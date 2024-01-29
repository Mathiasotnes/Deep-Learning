import matplotlib.pyplot as plt
import numpy as np
from utilities import calculate_time
from neural_network import Layer, Network, ReLU, Tanh, MSE

if __name__ == '__main__':
    # Generate dummy data for sine function
    X_train = np.linspace(-np.pi, np.pi, 100).reshape(1, 100)
    y_train = np.sin(X_train) + np.random.randn(*X_train.shape) * 0.1  # Sine function with noise

    # Create and train the network
    layer1 = Layer(1, 10, Tanh())
    layer2 = Layer(10, 2, Tanh())
    layer3 = Layer(2, 1, Tanh())
    network = Network([layer1, layer2, layer3], loss_function=MSE())
    network.fit(X_train, y_train, learning_rate=0.01, epochs=10000, verbose=1, batch_size=100)

    # Visualizing the results
    plt.figure(figsize=(10, 6))

    # Plot original sine data points
    plt.scatter(X_train, y_train, color='blue', label='Original data')

    # Generate predictions for plotting
    X_plot = np.linspace(-np.pi, np.pi, 100).reshape(1, 100)
    y_pred_plot = network.predict(X_plot)

    # Plot approximation
    plt.plot(X_plot.T, y_pred_plot.T, color='red', label='Network approximation')

    # Labeling the plot
    plt.xlabel('Input Feature')
    plt.ylabel('Target Value')
    plt.title('Sine Function Approximation with Neural Network')
    plt.legend()

    # Show plot
    plt.show()
