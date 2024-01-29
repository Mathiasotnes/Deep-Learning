import matplotlib.pyplot as plt
import numpy as np
from utilities import calculate_time
from neural_network import Layer, Network, Tanh, MSE, L1

if __name__ == '__main__':
    # Generate dummy data for sine function
    X_train = np.linspace(-np.pi, np.pi, 100).reshape(1, 100)
    y_train = np.sin(X_train) + np.random.randn(*X_train.shape) * 0.1  # Sine function with noise

    # Create and train the network
    layer1 = Layer(1, 5, Tanh())
    layer2 = Layer(5, 2, Tanh())
    layer3 = Layer(2, 1, Tanh())
    network = Network([layer1, layer2, layer3], loss_function=MSE(), regularization=L1(0.01))
    network.fit(X_train, y_train, learning_rate=0.01, epochs=100000, verbose=1, batch_size=100)

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
