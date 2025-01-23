"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library

import time
import math
import random

# Third-party libraries
import numpy as np

#### Miscellaneous functions
class ActivationFunction():
    def forward(self, z):
        raise NotImplementedError

    def derivative(self, z):
        raise NotImplementedError

    def __call__(self, z):
        return self.forward(z)

class Sigmoid(ActivationFunction):
    def forward(self, z):
        return 1.0/(1.0+np.exp(-z))

    def derivative(self, z):
        return self(z)*(1-self(z))

    def __repr__(self):
        return "Sigmoid"

# Not currently working.
class ReLU(ActivationFunction):
    def forward(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return (z > 0).astype(float)
    
class Tensor():
    def __init__(self, data):
        self.data = data


class Network(object):

    def __init__(self, sizes, activation_function='sigmoid'):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        self.activation_function = ReLU() if activation_function == 'relu' else Sigmoid()

        if activation_function == 'relu':
            print("Using He initialization for ReLU")
            # He initialization for ReLU
            # W ~ N(0, sqrt(2/n_in))
            for i in range(len(self.weights)):
                std = math.sqrt(2/self.weights[i].shape[1])  # shape[1] is n_in
                self.weights[i] = self.weights[i] * std
        else:
            print("Using Xavier/Glorot initialization for sigmoid/tanh")
            # #### Xavier/Glorot (for tanh/sigmoid)
            # - **Normal**: $W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in} + n_{out}}})$
            for i in range(len(self.weights)):
                std = math.sqrt(1/self.weights[i].shape[1])  # shape[1] is n_in
                self.weights[i] = self.weights[i] * std

    
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, test_slice=1000):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        print(f"Initial evaluation: {self.evaluate(test_data)} / {len(test_data)}")

        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for k, mini_batch in enumerate(mini_batches):
                self.update_mini_batch(mini_batch, eta)
                if k % 10 == 0:
                    print(f'[{k*mini_batch_size}/{n}]: {self.evaluate(test_data[:test_slice])} / {test_slice} correct')

            time2 = time.time()
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {len(test_data)}, took {time2-time1:.2f} seconds")
            else:
                print(f"Epoch {j} complete in {time2-time1:.2f} seconds")


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.activation_function.derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_function.derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    

# def sigmoid(z):
#     """The sigmoid function."""
#     return 1.0/(1.0+np.exp(-z))

# def sigmoid_prime(z):
#     """Derivative of the sigmoid function."""
#     return sigmoid(z)*(1-sigmoid(z))
