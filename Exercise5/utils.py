import numpy as np
import struct
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def load_svhn_data(path='./iannwtf-svhn/', train_data='trainData.pickle',
        train_labels='trainLabels.pickle', train_split=0.75, test_split=0.10,
        val_split=0.15, flatten=False, normalize=False):

    assert abs(train_split + test_split + val_split) - 1 < 1e-4, "Split sizes don't sum to 1"

    import pickle
    import numpy as np

    with open(path + train_data, mode='rb') as data_file:
        with open(path + train_labels, mode='rb') as label_file:
            # load the data
            d = pickle.load(data_file)
            l = pickle.load(label_file)
            if normalize:
                d = (d - np.mean(d)) / np.std(d)

            # first split train/test
            d_train, d_test, l_train, l_test = train_test_split(d, l, train_size = train_split, random_state = 0)
            # split test set further
            d_test, d_val, l_test, l_val = train_test_split(d_test, l_test,
                    train_size=test_split/(test_split + val_split),
                    random_state=0)
            if flatten:
                new_shape = (-1, d_train.shape[1] * d_train.shape[2], d_train.shape[3])
                d_train = np.reshape(d_train, new_shape)
                d_test  = np.reshape(d_test, new_shape)
                d_val   = np.reshape(d_val, new_shape)
            return d_train, l_train, d_test, l_test, d_val, l_val


def plot_data():
    d_train, l_train, _, _, _, _ = load_svhn_data()
    d_train = np.squeeze(d_train)
    for l in set(l_train):
        d_train_l = d_train[l_train == l,:,:]
        print("Samples for label {}: {}".format(l, d_train_l.shape[0] /
            d_train.shape[0]))

    d_train = (d_train - np.mean(d_train)) / np.std(d_train)
    f, axarr = plt.subplots(5,5)
    for k in range(25): 
        ax = axarr[k//5][k % 5]
        ax.imshow(d_train[k], cmap='gray', interpolation='nearest')
        ax.set_title(l_train[k], loc='left')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
    plt.show()
            
if __name__ == "__main__":
    plot_data()

def one_hot(vector, slots):
    arr = np.zeros((len(vector), slots))
    arr[range(len(vector)), vector - 1] = 1
    return arr;

def minibatches(data, labels, batch_size=1000):
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
                    data[batch:batch+batch_size,:],
                    one_hot(labels[batch:batch+batch_size], 10)
                    )
