from simplegrad import Tensor, Sequential, Linear, SGD, MnistLoader

Tensor.set_seed()

device = Tensor.get_device()
device = None
training_data, validation_data, test_data = MnistLoader.load_data_wrapper(parent_dir=True, mini=True)

training_data, validation_data, test_data = [[[Tensor(x, device=device), Tensor(y, device=device)] for x, y in data] 
                                             for data in [training_data, validation_data, test_data]]


Tensor.set_seed()
sizes = [784, 10, 10]
# sizes = [784, 5, 5, 10]
model = Sequential([
    Linear(in_size, out_size) 
            for in_size, out_size in zip(sizes[:-1], sizes[1:])])
# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)

SGD(model, training_data, epochs=1, mini_batch_size=3, eta=3.0, test_data=validation_data, test_interval=10)
