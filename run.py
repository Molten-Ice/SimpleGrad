from simplegrad import Tensor, Sequential, Linear, SGD, MnistLoader

Tensor.set_seed()

device = Tensor.get_device()
device = None
mini, test_interval, epochs = True, 50, 1
# mini, test_interval, epochs = False, 1000, 1
training_data, validation_data, test_data = MnistLoader.load_data_wrapper(parent_dir=True, mini=mini)

training_data, validation_data, test_data = [[[Tensor(x, device=device), Tensor(y, device=device)] for x, y in data] 
                                             for data in [training_data, validation_data, test_data]]

Tensor.set_seed()
sizes = [784, 100, 10]

# lr, loss_func, lambda_reg, epochs, dropout = 3, 'mse', 0, 1, 0 # Epoch 0: 819 / 1000, took 1.16 seconds
# lr, loss_func, lambda_reg , epochs, dropout = 0.5, 'cross_entropy', 0, 1, 0 # Epoch 0: 851 / 1000, took 1.42 seconds (base_loss: 1.307331, l2_loss: 0.020721)
# lr, loss_func, lambda_reg , epochs, dropout = 0.5, 'cross_entropy', 0.002, 1, 0 # Epoch 0: 840 / 1000, took 1.93 seconds (base_loss: 0.92043, l2_loss: 0.02926)
# lr, loss_func, lambda_reg , epochs, dropout = 1, 'cross_entropy', 0.002, 3, 0.3 # Epoch 2: 831 / 1000, took 5.92 seconds (base_loss: 1.79011, l2_loss: 0.05985)


# lr, loss_func, lambda_reg , epochs, dropout = 0.5, 'cross_entropy', 0, 5, 0 # Epoch 4: 909 / 1000, took 3.12 seconds ( base_loss: 0.09795, weight_sizes: [0.06274, 0.48611])
# lr, loss_func, lambda_reg , epochs, dropout = 0.5, 'cross_entropy', 3e-4, 5, 0 # Epoch 4: 904 / 1000, took 5.35 seconds (base_loss: 0.14461, l2_loss: 0.01248, weight_sizes: [0.05545, 0.44892])

lr, loss_func, lambda_reg , epochs, dropout = 0.5, 'cross_entropy', 3e-4, 1, 0 

model = Sequential([
    Linear(in_size, out_size, activation='sigmoid', dropout=dropout if i != len(sizes)-2 else 0) # activation='relu' if i != len(sizes)-2 else 'sigmoid'
            for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:]))])

SGD(model, training_data, epochs=epochs, mini_batch_size=10, lr=lr, test_data=validation_data, test_interval=test_interval, loss_func=loss_func, 
    lambda_reg=lambda_reg)

# TODO:
# add convolutional layers & maxpooling
# Speedup code (alot)