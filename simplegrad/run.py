from network import Sequential, Linear, Sigmoid, SGD, MSE
import mnist_loader as mnist_loader
from tensor import Tensor

Tensor.set_seed()

device = Tensor.get_device()
device = None
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(parent_dir=True, device=device, mini=True)

Tensor.set_seed()
sizes = [784, 10, 10]
# sizes = [784, 3, 3, 10]
net = Sequential([
    Linear(in_size, out_size, activation_function=Sigmoid()) 
            for in_size, out_size in zip(sizes[:-1], sizes[1:])] + [MSE()])
# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)

SGD(net, training_data, epochs=1, mini_batch_size=3, eta=3.0, test_data=validation_data, test_interval=10)

# Group into NN layers (That's how the error term from Michael was defined.)
# Create a graph based on NN layers.

# Store z and a for each layer.

# Create a topological sort of the graph.

# Start in reverse, backpropagate the error term. (also updating weights and biases at each Layer)

# Add requires_grad=False to input to stop gradient from being computed.

# weights, add bias, apply activation function

## Next steps ##
# Split it up into its components, working more similarly to pytorch, i.e. based on individual Tensors. 
# (Need to calculate grad w.r.t. weights, biases and activation function separately.)
