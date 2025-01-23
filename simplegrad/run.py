import network
import mnist_loader as mnist_loader
from tensor import Tensor

Tensor.set_seed()

device = Tensor.get_device()
device = None
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(parent_dir=True, device=device, mini=True)
net = network.Network([784, 10, 10]).to(device)
# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)

net.SGD(training_data, epochs=1, mini_batch_size=10, eta=3.0, test_data=validation_data, test_interval=10)
