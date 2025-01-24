from network import Network, Sequential, Linear, Sigmoid, evaluate, SGD, MSE
import mnist_loader as mnist_loader
from tensor import Tensor

Tensor.set_seed()

device = Tensor.get_device()
device = None
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(parent_dir=True, device=device, mini=True)
sizes = [784, 10, 10]

Tensor.set_seed()
net = Network(sizes).to(device)
# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)

SGD(net, training_data, epochs=1, mini_batch_size=10, eta=3.0, test_data=validation_data, test_interval=10, original=True)

# Building autograd type model structure.
# print(f"Initial evaluation: {evaluate(net, test_data)} / {len(test_data)}")
# net.update_mini_batch(training_data[:10], eta=3.0)
print(f"Net 1 evaluation: {evaluate(net, test_data)} / {len(test_data)}")

net2 = Sequential([
    Linear(in_size, out_size, activation_function=Sigmoid()) 
            for in_size, out_size in zip(sizes[:-1], sizes[1:])])

for net_layer, (w, b) in zip(net2.layers, zip(net.weights, net.biases)):
    net_layer.w = w
    net_layer.b = b

print(f"Net 2 evaluation: {evaluate(net2, test_data)} / {len(test_data)}")

Tensor.set_seed()
net3 = Sequential([
    Linear(in_size, out_size, activation_function=Sigmoid()) 
            for in_size, out_size in zip(sizes[:-1], sizes[1:])],
            loss_function=MSE())
SGD(net3, training_data, epochs=1, mini_batch_size=10, eta=3.0, test_data=validation_data, test_interval=10)
print(f"Net 3 evaluation: {evaluate(net3, test_data)} / {len(test_data)}")

# import sys; sys.exit()
net4 = Network(sizes)

for i, (net_layer, (w, b)) in enumerate(zip(net3.layers, zip(net4.weights, net4.biases))):
    net4.weights[i] = net_layer.w
    net4.biases[i] = net_layer.b


print(f"Net 4 evaluation: {evaluate(net4, test_data)} / {len(test_data)}")

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
