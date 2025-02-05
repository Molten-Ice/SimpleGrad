from simplegrad import Tensor, Sequential, Linear, SGD, MnistLoader

Tensor.set_seed()

device = Tensor.get_device()
device = None
mini, test_interval, epochs = True, 50, 1
mini, test_interval, epochs = False, 1000, 1
training_data, validation_data, test_data = MnistLoader.load_data_wrapper(parent_dir=True, mini=mini)

training_data, validation_data, test_data = [[[Tensor(x, device=device), Tensor(y, device=device)] for x, y in data] 
                                             for data in [training_data, validation_data, test_data]]


Tensor.set_seed()
sizes = [784, 100, 10]

# lr, loss_func, lambda_reg, epochs = 3, 'mse', 0, 1 # Epoch 0: 819 / 1000, took 6.08 seconds
lr, loss_func, lambda_reg , epochs, dropout = 0.5, 'cross_entropy', 0.01, 1, 0 # Epoch 0: 847 / 1000, took 3.30 seconds
lr, loss_func, lambda_reg , epochs, dropout = 0.5, 'cross_entropy', 0.01, 5, 0.5 # Epoch 4: 862 / 1000, took 3.44 seconds

model = Sequential([
    Linear(in_size, out_size, activation='sigmoid', dropout=dropout if i != len(sizes)-2 else 0) # activation='relu' if i != len(sizes)-2 else 'sigmoid'
            for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:]))])


SGD(model, training_data, epochs=epochs, mini_batch_size=10, lr=lr, test_data=validation_data, test_interval=test_interval, loss_func=loss_func, 
    lambda_reg=lambda_reg)

# TODO:
# add convolutional layers & maxpooling