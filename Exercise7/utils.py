import numpy as np
import struct
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MNIST():
    def __init__(self, directory = "./"):
        self.testData = self._load(directory + "t10k-images.idx3-ubyte")
        self.testLabels = self._load(directory + "t10k-labels.idx1-ubyte", True)
        self.trainingData = self._load(directory + "train-images.idx3-ubyte")
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

def normalize(data):
        return data / data.max()

def urand_vector(low=-1, high=1, shape=(1,100)):
    return np.random.uniform(low, high ,shape)

def one_hot(vector, slots):
    arr = np.zeros((len(vector), slots))
    arr[np.arange(len(vector)), vector] = 1
    return arr;

def minibatches(data, labels, batch_size=1000, loop=False):
    """
    Return randomly shuffled minibatches
    """
    assert data.shape[0] == len(labels)
    while True:
        indices = np.random.permutation(data.shape[0])
        data = data[indices, :]
        labels = labels[indices]
        for batch in np.arange(0, data.shape[0], batch_size):
            yield (
                    data[batch:batch+batch_size,:,:],
                    one_hot(labels[batch:batch+batch_size], 10)
                    )
        if not loop:
            break
