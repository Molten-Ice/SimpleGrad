import network
import utils.mnist_loader as mnist_loader


# Set random seeds for reproducibility
import random
import numpy as np
np.random.seed(3)
random.seed(3)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# MNIST data in form (50000, 2), (10000, 2), (10000, 2)
# training_data[0][0].shape -> (784, 1), training_data[0][1].shape -> (10, 1)

# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)
net = network.Network([784, 10, 10])

# net.SGD(training_data[:3000], epochs=1, mini_batch_size=10, eta=3.0, test_data=test_data)

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")