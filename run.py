from simplegrad import Tensor, Sequential, Linear, SGD, MnistLoader

Tensor.set_seed()

device = 'cpu' # Tensor.get_device() # quicker than gpu for some reason
mini, test_interval, epochs = True, 50, 1
mini, test_interval, epochs = False, 1000, 1
training_data, validation_data, test_data = MnistLoader.load_data_wrapper(parent_dir=True, mini=mini)


training_data, validation_data, test_data = [[[Tensor(x, device=device), Tensor(y, device=device)] for x, y in data] 
                                             for data in [training_data, validation_data, test_data]]

Tensor.set_seed()
sizes = [784, 100, 10]

lr, loss_func, lambda_reg, epochs, dropout = 0.5, 'cross_entropy', 3e-4, 1, 0.0

model = Sequential([
    Linear(in_size, out_size, activation='sigmoid', dropout=dropout if i != len(sizes)-2 else 0) # activation='relu' if i != len(sizes)-2 else 'sigmoid'
            for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:]))])
model.to(device)

# SGD(model, training_data, epochs=epochs, batch_size=10, lr=lr, test_data=validation_data, test_interval=test_interval, loss_func=loss_func,  lambda_reg=lambda_reg)

import cProfile
cProfile.run('SGD(model, training_data, epochs=epochs, batch_size=10, lr=lr, test_data=validation_data, test_interval=test_interval, loss_func=loss_func,  lambda_reg=lambda_reg)', 'output.stats')

#snakeviz output.stats

# TODO:
# add convolutional layers & maxpooling
# Speedup code (alot)

# Epoch 0: 9514 / 10000, took 20.36 seconds