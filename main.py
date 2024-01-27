import matplotlib.pyplot as plt
import numpy as np
from utilities import calculate_time
import models
import activations
import losses

if __name__ == '__main__':
    # Generate dummy data
    X_train = np.linspace(-1, 1, 100).reshape(1, 100)
    y_train = 2 * X_train + 3 + np.random.randn(*X_train.shape) * 0.2

    # Create and train the network
    layer1 = models.Layer(1, 3, activations.ReLU(), losses.MSE())
    layer2 = models.Layer(3, 2, activations.ReLU(), losses.MSE())
    layer3 = models.Layer(2, 1, activations.ReLU(), losses.MSE())
    network = models.Network([layer1, layer2, layer3])
    network.fit(X_train, y_train, learning_rate=0.001, epochs=10000)

    # Visualizing the results
    plt.figure(figsize=(10, 6))

    # Plot original data points
    plt.scatter(X_train, y_train, color='blue', label='Original data')

    # Generate predictions for plotting
    X_plot = np.linspace(-1, 1, 100).reshape(1, 100)
    y_pred_plot = network.predict(X_plot)

    # Plot regression line
    plt.plot(X_plot.T, y_pred_plot.T, color='red', label='Fitted line')

    # Labeling the plot
    plt.xlabel('Input Feature')
    plt.ylabel('Target Value')
    plt.title('Linear Regression with Neural Network')
    plt.legend()

    # Show plot
    plt.show()
