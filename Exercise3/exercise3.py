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


if __name__ == "__main__":
    d_train, l_train, d_test, l_test, d_val, l_val = import_data()
    batch_size = 100

    # plot_some_digits(d_train, l_train)
    W = tf.Variable(tf.zeros([28 * 28, 10]))
    b = tf.Variable(tf.zeros([10]))
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    d = tf.placeholder(tf.int32, [None, 10])
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    # cost = tf.reduce_sum(tf.mul(tf.constant(0.5), tf.square(tf.sub(d, y))))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b, d)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5) 
    minimizer = optimizer.minimize(cross_entropy)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for iteration in range(10):
            for mb, labels in minibatches(d_train, l_train, batch_size=batch_size):
                minimizer.run(feed_dict={x: mb, d: labels})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("accuracy: %f" % accuracy.eval({x: d_test, d: one_hot(l_test, 10)}))

