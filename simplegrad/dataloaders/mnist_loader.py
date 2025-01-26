import os
import gzip
import pickle
# Implicitly using numpy as that's the imported data type.


class MnistLoader:

    @staticmethod
    def load_data_wrapper(parent_dir=False, slice_training=None, slice_test=None, mini=False):
        """Returns (training_data, validation_data, test_data) where training_data contains 50,000 (784x1, 10x1) tuples,
        and validation_data/test_data each contain 10,000 (784x1, 1x1) tuples.
        Each 784-dim vector is a flattened 28x28 MNIST image, and the output dimensions differ for training vs validation/test."""
        tr_d, va_d, te_d = MnistLoader._load_data(parent_dir, slice_training, slice_test, mini)
        training_inputs = [x.reshape(784, 1) for x in tr_d[0]]
        validation_inputs = [x.reshape(784, 1) for x in va_d[0]]
        test_inputs = [x.reshape(784, 1) for x in te_d[0]]
        
        training_results = [MnistLoader._vectorized_result(y) for y in tr_d[1]]
        training_data = list(zip(training_inputs, training_results))
        validation_data = list(zip(validation_inputs, [[y] for y in va_d[1]]))
        test_data = list(zip(test_inputs, [[y] for y in te_d[1]]))

        return (training_data, validation_data, test_data)


    @staticmethod
    def _load_data(parent_dir=False, slice_training=None, slice_test=None, mini=False):
        """Returns (training_data, validation_data, test_data) where training_data is a tuple of (images, labels).
        Images are numpy arrays of 50,000 x 784 (flattened 28x28 pixels) for training, 10,000 x 784 for validation/test.
        Labels are numpy arrays of corresponding digits (0-9) with lengths 50,000 (training) or 10,000 (validation/test)."""
        parent_dir_string = '' if parent_dir else '../'

        original_file = f'{parent_dir_string}data/mnist.pkl.gz'
        assert os.path.exists(original_file), f"File {original_file} does not exist"

        base_path = 'mini_mnist.pkl.gz' if mini else 'mnist.pkl.gz'
        path = f'{parent_dir_string}data/{base_path}'

        if mini and not os.path.exists(path):
            print('Creating mini MNIST dataset...')
            training_data, validation_data, test_data = MnistLoader._create_mini_mnist(output_path=path, parent_dir=parent_dir)
        else:
            print(f'Loading MNIST {"full" if not mini else "mini"} dataset...')

            f = gzip.open(path, 'rb')
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            training_data, validation_data, test_data = u.load()
            f.close()

        if slice_training:
            training_data = (training_data[0][:slice_training], training_data[1][:slice_training])

        if slice_test:
            test_data = (test_data[0][:slice_test], test_data[1][:slice_test])
            validation_data = (validation_data[0][:slice_test], validation_data[1][:slice_test])

        return (training_data, validation_data, test_data)
    
    @staticmethod
    def _create_mini_mnist(output_path='data/mini_mnist.pkl.gz', training_size=3000, eval_test_size=1000, parent_dir=False):
        # Load the full dataset with specified slices
        training_data, validation_data, test_data = MnistLoader._load_data(
            slice_training=training_size,
            slice_test=eval_test_size,
            parent_dir=parent_dir
        )

        with gzip.open(output_path, 'wb') as f:
            pickle.dump((training_data, validation_data, test_data), f)
        return (training_data, validation_data, test_data)

    @staticmethod
    def _vectorized_result(j):
        """Returns a 10x1 unit vector with 1.0 at index j and zeros elsewhere, converting a digit (0-9) to one-hot encoding."""
        e = [[0.0] for _ in range(10)]
        e[j][0] = 1.0
        return e