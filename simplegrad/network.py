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

        std = math.sqrt(1/input_size)  # shape[1] is n_in
        self.w = std *self.w
    
    def forward(self, x):
        out = self.w @ x + self.b
        if self.activation_function:
            out = self.activation_function(out)
        return out
    
    def backward(self, x):
        return x
        # return self.activation_function.derivative(self.w @ x + self.b) * self.w

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def SGD(net, training_data, epochs, mini_batch_size, eta,
        test_data=None, test_interval=None):
    """Train the neural network using mini-batch stochastic
    gradient descent.  The ``training_data`` is a list of tuples
    ``(x, y)`` representing the training inputs and the desired
    outputs.  The other non-optional parameters are
    self-explanatory.  If ``test_data`` is provided then the
    network will be evaluated against the test data after each
    epoch, and partial progress printed out.  This is useful for
    tracking progress, but slows things down substantially."""

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
            # Zero gradients (optimizer.zero_grad())

            # Backpropagate (loss.backward())
            nabla_b, nabla_w = net.backprop(xb, yb)

            # Update weights and biases (optimizer.step())
            net.weights = [w-scaled_eta*nw for w, nw in zip(net.weights, nabla_w)]
            net.biases = [b-scaled_eta*nb for b, nb in zip(net.biases, nabla_b)]

            if test_interval is not None and k % test_interval == 0:
                print(f'[{k*mini_batch_size}/{n}]: {evaluate(net, test_data, batch_size=test_interval)} / {len(test_data)} correct')

        if test_data:
            print(f"Epoch {j}: {evaluate(net, test_data)} / {len(test_data)}, took {time.time()-time1:.2f} seconds")
        else:
            print(f"Epoch {j} complete in {time.time()-time1:.2f} seconds")



class Network(object):

    def __init__(self, sizes):
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


        self.biases = [Tensor.randn(y, 1) for y in sizes[1:]]
        self.weights = [Tensor.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.activation_function = Sigmoid()

        self.nabla_b = [Tensor.zeros(b.shape) for b in self.biases]
        self.nabla_w = [Tensor.zeros(w.shape) for w in self.weights] 

        # Xavier/Glorot (for tanh/sigmoid)
        # - **Normal**: $W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in} + n_{out}}})$
        for i in range(len(self.weights)):
            std = math.sqrt(1/self.weights[i].shape[1])  # shape[1] is n_in
            self.weights[i] = std *self.weights[i]

    
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(w @ a + b)
        return a
    
    def __call__(self, x):
        return self.feedforward(x)
        

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [Tensor.zeros(b.shape) for b in self.biases]
        nabla_w = [Tensor.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights): # 2 layers
            z = w @ activation + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

            
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.activation_function.derivative(zs[-1])
        nabla_b[-1] = delta

        # xb.shape: torch.Size([10, 784, 1]), yb.shape: torch.Size([10, 10, 1]) 
        # Original
        # delta.shape: torch.Size([10, 1]), activations[-2].shape: torch.Size([10, 1])
        # activations[-2].transpose().shape: torch.Size([1, 10])
        # Batch:
        # delta.shape: torch.Size([10, 10, 1]), activations[-2].shape: torch.Size([10, 10, 1])
        # activations[-2].transpose().shape: torch.Size([10, 10, 1])

        nabla_w[-1] = delta @ activations[-2].transpose(-2, -1)
        # nabla_w[-1] = delta @ activations[-2].transpose()
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, len(self.weights)+1):
            z = zs[-l]
            sp = self.activation_function.derivative(z)
            delta = self.weights[-l+1].transpose() @ delta * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l-1].transpose(-2, -1)

        return ([x.sum(dim=0) for x in nabla_b], [x.sum(dim=0) for x in nabla_w])
    

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    

    def to(self, device):
        """Move all network parameters to specified device (e.g. 'cuda' or 'cpu')"""
        print(f"Moving model to device: {device}")
        self.device = device
        if device:
            self.weights = [w.to(device) for w in self.weights]
            self.biases = [b.to(device) for b in self.biases]
        return self


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