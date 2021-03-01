"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data():
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
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Manipulating data to a convenient form.
    """
    tr_d, va_d, te_d = load_data()

    #plot some numbers!
    #plt.show()
    #data = [np.reshape(x, (28,28)) for x in tr_d[0]]
    #for i in range(6):
    #    plt.imshow(data[i], cmap='gray')
    #    plt.show(block=True)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = tr_d[1]
    test_results = te_d[1]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    n_components = 50
    pca = PCA(n_components)
    x_train = np.hstack(training_inputs)
    pca.fit(x_train.transpose())
    x_transformed_train = pca.transform(x_train.transpose())
    training_inputs = x_transformed_train
    x_test = np.hstack(test_inputs)
    x_transformed_test = pca.transform(x_test.transpose())
    test_inputs = x_transformed_test

    return (training_inputs, training_results), (test_inputs, test_results), tr_d[0]
