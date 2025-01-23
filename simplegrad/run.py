import network
import utils.mnist_loader as mnist_loader


# Set random seeds for reproducibility
import random
import numpy as np
np.random.seed(3)
random.seed(3)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper(parent_dir=True)
# MNIST data in form (50000, 2), (10000, 2), (10000, 2)
# training_data[0][0].shape -> (784, 1), training_data[0][1].shape -> (10, 1)

# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)
net = network.Network([784, 10, 10])

net.SGD(training_data[:3000], epochs=1, mini_batch_size=10, eta=3.0, test_data=test_data)

import torch

# Basic CUDA check
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'None'}")

# Test CUDA with a simple tensor operation
if torch.cuda.is_available():
    # Create a tensor on GPU
    x = torch.rand(5, 3).cuda()
    print(f"Tensor device: {x.device}")
    print("CUDA is working properly!")
else:
    print("CUDA is not available")