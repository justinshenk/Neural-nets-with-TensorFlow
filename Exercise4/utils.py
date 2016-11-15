import numpy as np
import struct


class MNIST():
    def __init__(self, directory = "./"):
        self.testData = np.reshape(self._load(directory + "t10k-images.idx3-ubyte"), (-1,28**2))
        self.testLabels = self._load(directory + "t10k-labels.idx1-ubyte", True)
        self.trainingData = np.reshape(self._load(directory + "train-images.idx3-ubyte"), (-1,28**2))
        self.trainingLabels = self._load(directory + "train-labels.idx1-ubyte", True)

    def _load(self, path, labels = False):

        with open(path, "rb") as fd:
            magic, numberOfItems = struct.unpack(">ii", fd.read(8))
            if (not labels and magic != 2051) or (labels and magic != 2049):
                raise LookupError("Not a MNIST file")

            if not labels:
                rows, cols = struct.unpack(">II", fd.read(8))
                images = np.fromfile(fd, dtype = 'uint8')
                images = images.reshape((numberOfItems, rows, cols))
                return images
            else:
                labels = np.fromfile(fd, dtype = 'uint8')
                return labels

def load_mnist_data():
    n_train = 55000
    m                 = MNIST()
    data_train        = m.trainingData[:n_train]
    labels_train      = m.trainingLabels[:n_train]

    data_test         = m.testData
    labels_test       = m.testLabels
    data_validation   = m.trainingData[n_train:]
    labels_validation = m.trainingLabels[n_train:]

    return data_train, labels_train, data_test, labels_test, data_validation, labels_validation

def load_mnist_data_tf():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=False)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def one_hot(vector, slots):
    arr = np.zeros((len(vector), slots))
    arr[range(len(vector)), vector] = 1
    return arr;

def minibatches(data, labels, batch_size=1000):
    """Data must be in Nx784 shape.
    Return randomly shuffled minibatches
    """
    assert data.shape[0] == len(labels)
    indices = np.random.permutation(data.shape[0])
    data = data[indices, :]
    labels = labels[indices]
    for batch in np.arange(0, data.shape[0], batch_size):
        if batch + batch_size > data.shape[0]: # if data size does not divide evenly, make final batch smaller
            batch_size = data.shape[0] - batch
        yield (
                data[batch:batch+batch_size,:],
                one_hot(labels[batch:batch+batch_size], 10)
                )
