import numpy as np
import tensorflow as tf
from numpy.random import random_integers
import matplotlib.pyplot as plt
import struct

class MNIST():
    def __init__(self, directory = "./"):
        self.testData = np.reshape(self._load(directory + "t10k-images-idx3-ubyte"), (-1,28**2))
        self.testLabels = self._load(directory + "t10k-labels-idx1-ubyte", True)
        self.trainingData = np.reshape(self._load(directory + "train-images-idx3-ubyte"), (-1,28**2))
        self.trainingLabels = self._load(directory + "train-labels-idx1-ubyte", True)
    
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

def import_data(split=2):
    m                 = MNIST()
    data_train        = m.trainingData
    labels_train      = m.trainingLabels

    # make a 50/50 test-validation split
    n_test            = len(m.testData)
    data_test         = m.testData[:n_test//split]
    labels_test       = m.testLabels[n_test//split:]
    data_validation   = m.testData[n_test//split:]
    labels_validation = m.testLabels[n_test//split:]

    return data_train, labels_train, data_test, labels_test, data_validation, labels_validation

def plot_some_digits(d_train, l_train):
    f, axarr = plt.subplots(5,2)
    f.tight_layout()
    f.subplots_adjust(hspace=.05, wspace=.05)
    axarr_linear = np.reshape(axarr, 10)
    plt.setp([a.get_xticklabels() for a in axarr_linear], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr_linear], visible=False)

    indices = random_integers(0, d_train.shape[0], size=10)
    for i, index in enumerate(indices):
        axarr_linear[i].imshow(np.reshape(d_train[index,:], (28,28)), cmap='gray')
        axarr_linear[i].text(-10,20,"{}".format(l_train[index]))
    plt.show()
    

if __name__ == "__main__":
    d_train, l_train, d_test, l_test, d_val, l_val = import_data()

    plot_some_digits(d_train, l_train)
    W = tf.Variable(tf.random_normal([28*28,10]))
    b = tf.Variable(tf.random_normal([10,1]), trainable=False)
    x = tf.placeholder(tf.float32, [1,28*28])
    y = tf.nn.softmax(tf.add(tf.matmul(x,W), b))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        out = sess.run(y, feed_dict={x: np.reshape(d_train[0,:,:], (-1,28*28))})

