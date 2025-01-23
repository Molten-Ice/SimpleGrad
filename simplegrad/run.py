import network
import mnist_loader as mnist_loader
from tensor import Tensor

Tensor.set_seed()

training_data, validation_data, test_data = mnist_loader.load_data_wrapper(parent_dir=True)
# MNIST data in form (50000, 2), (10000, 2), (10000, 2)
# training_data[0][0].shape -> (784, 1), training_data[0][1].shape -> (10, 1)

# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'None'}")


net = network.Network([784, 10, 10]).to(device)

net.SGD(training_data[:3000], epochs=1, mini_batch_size=10, eta=3.0, test_data=test_data)

# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'None'}")
