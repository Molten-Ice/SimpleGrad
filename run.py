from simplegrad import Tensor, Sequential, Linear, SGD, MnistLoader

Tensor.set_seed()

device = Tensor.get_device()
device = None
mini = True
# mini = False
training_data, validation_data, test_data = MnistLoader.load_data_wrapper(parent_dir=True, mini=mini)
training_data, validation_data, test_data = [[[Tensor(x, device=device), Tensor(y, device=device)] for x, y in data] 
                                             for data in [training_data, validation_data, test_data]]


Tensor.set_seed()
sizes = [784, 10, 10]
# sizes = [784, 5, 5, 10]

model = Sequential([
    Linear(in_size, out_size, activation='relu' if i != len(sizes)-2 else 'sigmoid') 
            for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:]))])
print(model)

eta = 0.3
SGD(model, training_data, epochs=1, mini_batch_size=10, eta=eta, test_data=validation_data, test_interval=10)


model = Sequential([
    Linear(in_size, out_size, activation='sigmoid') 
            for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:]))])
# - 784 input neurons (28x28 pixels),  3 neurons in hidden layer, 10 output neurons (digits 0-9)

eta = 3
SGD(model, training_data, epochs=1, mini_batch_size=10, eta=eta, test_data=validation_data, test_interval=10)

# TODO:
# Create SVM classifier
# add cross-entropy cost function,
# add regularization,
# add convolutional, maxpooling,
# add tanh