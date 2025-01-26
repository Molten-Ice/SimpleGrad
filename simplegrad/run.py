from tensor import Tensor
import mnist_loader as mnist_loader

from engine import Sequential, Linear, SGD

Tensor.set_seed()

device = Tensor.get_device()
device = None
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(parent_dir=True, device=device, mini=True)

Tensor.set_seed()
sizes = [784, 10, 10]
# sizes = [784, 3, 3, 10]
model = Sequential([
    Linear(in_size, out_size) 
            for in_size, out_size in zip(sizes[:-1], sizes[1:])])
# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)

SGD(model, training_data, epochs=1, mini_batch_size=3, eta=3.0, test_data=validation_data, test_interval=10)

