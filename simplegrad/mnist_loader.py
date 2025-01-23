"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

import time
import gzip
import pickle
from tensor import Tensor

def load_data(parent_dir=False, slice_training=None, slice_test=None, mini=False):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.


    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    parent_dir_string = '' if parent_dir else '../'
    base_path = 'mini_mnist.pkl.gz' if mini else 'mnist.pkl.gz'
    f = gzip.open(f'{parent_dir_string}data/{base_path}', 'rb')
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


def create_mini_mnist(output_path='data/mini_mnist.pkl.gz', training_size=3000, eval_test_size=1000, parent_dir=False):
    # Load the full dataset with specified slices
    training_data, validation_data, test_data = load_data(
        slice_training=training_size,
        slice_test=eval_test_size,
        parent_dir=parent_dir
    )

    with gzip.open(output_path, 'wb') as f:
        pickle.dump((training_data, validation_data, test_data), f)


def load_data_wrapper(parent_dir=False, device=None, slice_training=None, slice_test=None, mini=False):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data(parent_dir, slice_training, slice_test, mini)
    training_inputs = [Tensor(x, device=device).reshape(784, 1) for x in tr_d[0]]
    validation_inputs = [Tensor(x, device=device).reshape(784, 1) for x in va_d[0]]
    test_inputs = [Tensor(x, device=device).reshape(784, 1) for x in te_d[0]]
    
    training_results = [vectorized_result(y, device=device) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_data = list(zip(validation_inputs, [Tensor([y], device=device) for y in va_d[1]]))
    test_data = list(zip(test_inputs, [Tensor([y], device=device) for y in te_d[1]]))

    return (training_data, validation_data, test_data)

def vectorized_result(j, device=None):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = Tensor.zeros(10, 1, device=device)
    e.data[j] = 1.0
    return e
