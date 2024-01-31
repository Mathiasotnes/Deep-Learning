import matplotlib.pyplot as plt
import numpy as np
from neural_network import Layer, Network, Tanh, ReLU, MSE, L1
from data_generation import Generator

if __name__ == '__main__':

    generator = Generator(image_size=50, flatten=True, total_images=100)
    train_img, val_img, test_img = generator.generate(noise_level=0)
    # generator.visualize_images(train_img[0][:16], train_img[1][:16])
    
    # Define Layers
    layers = [
        Layer(50 * 50, 100, activation_function=Tanh()),
        Layer(100, 50, activation_function=Tanh()),
        Layer(50, 25, activation_function=ReLU()),
        Layer(25, 4, activation_function=Tanh())
    ]

    # Define regularization
    reg = L1(0.02)

    # Define Network
    network = Network(layers, loss_function=MSE(), regularization=reg)

    # Train Network
    network.fit(train_img[0],
                train_img[1],
                learning_rate=0.05,
                epochs=10000,
                verbose=1,
                batch_size=700)
    
    # Evaluate Network
    y_pred = network.predict(test_img[0])
    loss = network.loss_function.calculate(test_img[1], y_pred)
    print('Loss:', loss)
    print('Accuracy:', np.mean(np.argmax(y_pred, axis=0) == test_img[1]))

    # Visualize Predictions
    generator.visualize_images(test_img[0][:16], np.argmax(y_pred, axis=0)[:16])
    