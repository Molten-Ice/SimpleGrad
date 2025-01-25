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
from tensor import Tensor


#### Miscellaneous functions
class ActivationFunction():
    def __init__(self):
        self.func_type = 'activation'

    def forward(self, z):
        raise NotImplementedError

    def derivative(self, z):
        raise NotImplementedError

    def __call__(self, z):
        return self.forward(z)


class Sigmoid(ActivationFunction):
    def forward(self, z):
        return 1.0/(1.0 + (-z).exp())

    def derivative(self, z):
        return self(z)*(1-self(z))

    def __repr__(self):
        return "Sigmoid"
    
class Module():
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, x):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, input_size, output_size, activation_function=None):
        self.w = Tensor.randn(output_size, input_size)
        self.b = Tensor.randn(output_size, 1)
        self.activation_function = activation_function
        self.func_type = 'linear'


        # Really shoud be in a metadata class.
        self.metadata = {
            'nabla_w': Tensor.zeros(self.w.shape),
            'nabla_b': Tensor.zeros(self.b.shape),
            'delta': None,
            'z': None,
            'a': None,
        }

        std = math.sqrt(1/input_size)  # shape[1] is n_in
        self.w = std *self.w
    
    def forward(self, x):
        z = self.w @ x + self.b
        self.metadata['z'] = z
        if self.activation_function:
            a = self.activation_function(z)
        self.metadata['a'] = a
        return a


    def zero_grad(self):
        self.metadata['nabla_w'] *= 0
        self.metadata['nabla_b'] *= 0
        self.metadata['delta'] = None

    def step(self, eta):
        self.w -= eta * self.metadata['nabla_w']
        self.b -= eta * self.metadata['nabla_b']

    def backward(self, x):
        pass


class LossFunction():
    def __call__(self, output, target):
        return self.loss(output, target)

class MSE(LossFunction):
    def __init__(self):
        self.metadata = {
            'derivative': None
        }

    def loss(self, output, target):
        """Calculate Mean Squared Error loss"""
        self.metadata['derivative'] = self.derivative(output, target)
        return 0.5 * ((output - target) ** 2).mean()
    
    def derivative(self, output, target):
        return (output-target)


class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, y=None):
        for layer in self.layers[:-1]:
            x = layer(x)
        loss_function = self.layers[-1]
        if y is None:
            return x
        loss = loss_function(x, y)
        return x, loss


    def backward(self, x):
        """
w ─┐
   @ ─┐
x ─┘  │
      + ── z ── φ ── a
b ────┘
        """
        for l in range(2, len(self.layers)+1):
            # x1. Backpropagate to previous layer
            if l == 2: # Base case.
                loss_function = self.layers[-1]
                grad = loss_function.metadata['derivative']
            else:
                next_layer = self.layers[-l+1]
                grad = next_layer.w.transpose(-2, -1) @ grad # Shape remains unchanged.

            # x2. Backpropagate through activation function
            layer = self.layers[-l]
            grad *= layer.activation_function.derivative(layer.metadata['z'])

            # x3a. Backpropagate to bias (leaf node)
            layer.metadata['nabla_b'] += grad.sum(dim=0) # sum(dim=0) squashes batches.

            # x3b. Backpropagate to weights (leaf node)
            activation = x if l == len(self.layers) else self.layers[-l-1].metadata['a']
            layer.metadata['nabla_w'] += (grad @  activation.transpose(-2, -1)).sum(dim=0)


    def zero_grad(self):
        for layer in self.layers[:-1]:
            layer.zero_grad()

    def step(self, eta):
        for layer in self.layers[:-1]:
            layer.step(eta)


def SGD(net, training_data, epochs, mini_batch_size, eta,
        test_data=None, test_interval=None, original=False):
    print(f"Initial evaluation: {evaluate(net, test_data)} / {len(test_data)}")

    n = len(training_data)
    for j in range(epochs):
        time1 = time.time()
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]
        for k, mini_batch in enumerate(mini_batches):
            xb, yb = map(lambda t: Tensor.stack(t, dim=0), zip(*mini_batch))
            mini_batch_size = xb.shape[0]
            scaled_eta = eta/len(mini_batch)

            net.zero_grad() # Should be optimizer

            logits, loss = net.forward(xb, yb)
            # print(f"Loss: {loss.data:.5f}")

            net.backward(xb) # Should be loss.backward() not net.backward(logits)
            net.step(scaled_eta)

            if test_interval is not None and k % test_interval == 0:
                print(f'[{k*mini_batch_size}/{n}]: {evaluate(net, test_data, batch_size=test_interval)} / {len(test_data)} correct')

        if test_data:
            print(f"Epoch {j}: {evaluate(net, test_data)} / {len(test_data)}, took {time.time()-time1:.2f} seconds")
        else:
            print(f"Epoch {j} complete in {time.time()-time1:.2f} seconds")



def evaluate(net, test_data, batch_size=128):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation.
    
    Args:
        test_data: List of (x, y) tuples containing test inputs and labels
        batch_size: Size of batches to process at once (default: 32)
    """
    
    correct = 0
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        x = Tensor.stack([x for x, _ in batch])
        y = Tensor.stack([y for _, y in batch]).squeeze(dim=1).tolist()
        
        outputs = net(x)
        predictions = outputs.argmax(dim=1).squeeze(dim=1).tolist()
        correct += sum(int(pred == label) for pred, label in zip(predictions, y))
        
    return correct