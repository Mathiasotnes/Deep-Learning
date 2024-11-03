Deep-Learning
=============

Implementation of simple back-propagation using numpy.

Installation
------------
You can install `Deep-Learning` using pip:

```bash
pip install git+https://github.com/Mathiasotnes/Deep-Learning.git
```

Usage
-----

Quickly set up a neural network with multiple layers, including a softmax output layer, using the Deep Learning Library.

### Example: Multi-Layer Network with Softmax Output

```python
import numpy as np
from brain_of_mathias.models import Layer, Network
from brain_of_mathias.activations import ReLU, Softmax
from brain_of_mathias.losses import MSE

# Sample data - replace with actual data
X_train = np.array([...])  # Input features
y_train = np.array([...])  # Target labels

# Define a network with desired layers
layer1 = Layer(input_size=..., number_of_neurons=..., activation=ReLU())
layer2 = Layer(input_size=..., number_of_neurons=..., activation=ReLU())
output_layer = Layer(input_size=..., number_of_neurons=..., activation=Softmax())

# Initialize the network with the layers
network = Network([layer1, layer2, output_layer], loss_function=MSE())

# Train the network
network.fit(X_train, y_train, learning_rate=0.01, epochs=500)

# Predict
network.predict(X_test)
```

Features
--------
- Custom activation and loss functions.
- Extensible model architecture.
- Utilities for common operations.

Repo Activity
-------------
![Alt](https://repobeats.axiom.co/api/embed/20c237ee2eb3e404e339facea0ea8f99070ab15e.svg "Repobeats analytics image")
