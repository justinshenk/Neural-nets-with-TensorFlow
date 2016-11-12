import numpy as np
import tensorflow as tf
from numpy.random import random_integers
import matplotlib.pyplot as plt
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

def import_data():
    n_train = 55000
    m                 = MNIST()
    data_train        = m.trainingData[:n_train]
    labels_train      = m.trainingLabels[:n_train]

    data_test         = m.testData
    labels_test       = m.testLabels
    data_validation   = m.trainingData[n_train:]
    labels_validation = m.trainingLabels[n_train:]

    return data_train, labels_train, data_test, labels_test#, data_validation, labels_validation

def import_data_tf():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=False)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def plot_weights(current_weights):
    f, axarr = plt.subplots(2,5)
    f.tight_layout()
    f.subplots_adjust(hspace=.05, wspace=.05)
    axarr_linear = np.reshape(axarr, 10)
    plt.setp([a.get_xticklabels() for a in axarr_linear], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr_linear], visible=False)

    for col in range(current_weights.shape[1]):
        weights = np.reshape(current_weights[:,col], (28,28))
        axarr_linear[col].imshow(weights, cmap='seismic')

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
<<<<<<< HEAD
    plt.show()

=======
    
>>>>>>> 1c4dc387124473e1c1bb5fcc07c8fa2f2393237b
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


def main():

    # load the data
    d_train, l_train, d_test, l_test = import_data()
    # d_train_tf, l_train_tf, d_test_tf, l_test_tf = import_data_tf()
    # print("Are datasets equal? {}".format((d_train == d_train_tf).all()))

    batch_size = 1000

    # plot_some_digits(d_train, l_train)

    # Weight matrix
    W = tf.Variable(tf.zeros([28 * 28, 10]))

    # bias vector
    b = tf.Variable(tf.zeros([10]))

    # data vector
    x = tf.placeholder(tf.float32, [None, 28 * 28])

    # desired output (ie real labels)
    d = tf.placeholder(tf.int32, [None, 10])

    # computed output of the network without activation (?? otherwise can't use
    # tf.nn.softmax_..). Does this only work because there is just one layer?
    y = tf.matmul(x,W) + b

    # loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, d)
    optimizer     = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    training_step = optimizer.minimize(cross_entropy)

    # check if neuron firing strongest coinceds with max value position in real
    # labels
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # plot_some_digits(d_train, l_train)
    training_step_accuracy = []
    # validation_accuracy = []

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print("Training on my data.")
<<<<<<< HEAD
        for _ in range(1000):
            for mb, labels in minibatches(d_train, l_train, batch_size=batch_size):
                sess.run(training_step, feed_dict={x: mb, d: labels})
            current_accuracy = sess.run(accuracy, feed_dict={x: d_train, d: one_hot(l_train, 10)})
            training_step_accuracy.append(current_accuracy)
            # current_validation_accuracy = sess.run(accuracy, feed_dict={x: d_test, d: one_hot(l_test, 10)})
            # validation_accuracy.append(current_validation_accuracy)
=======
        for i in range(50):
            for mb, labels in minibatches(d_train, l_train, batch_size=batch_size):
                sess.run(training_step, feed_dict={x: mb, d: labels})
            if i > 1 and i % 10 == 0:
                current_weights = W.eval()
                plot_weights(current_weights)

        plt.show()
>>>>>>> 1c4dc387124473e1c1bb5fcc07c8fa2f2393237b
        print("accuracy: %f" % sess.run(accuracy, feed_dict={x: d_test, d: one_hot(l_test, 10)}))


    plt.plot(training_step_accuracy, color = "b")
    # plt.plot(validation_accuracy, color = "r")
    plt.show()

    # with tf.Session() as sess:
    #     sess.run(tf.initialize_all_variables())
    #     print("Training on tf data.")
    #     for _ in range(30):
    #         for mb, labels in minibatches(d_train_tf, l_train_tf, batch_size=batch_size):
    #             sess.run(training_step, feed_dict={x: mb, d: labels})
    #     print("accuracy: %f" % sess.run(accuracy, feed_dict={x: d_test_tf, d: one_hot(l_test_tf, 10)}))

if __name__ == "__main__":
    main()
