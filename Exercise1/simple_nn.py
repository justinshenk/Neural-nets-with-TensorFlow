# coding=utf-8

__author__    = "Rasmus Diederichsen, Kevin Trebing"
__date__      = "2016-11-05"
__email__     = "rdiederichse@uos.de"

"""We only implement stochastic gradient descent for time reasons and no
momentum."""

import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)

def logistic(z, derive=False):
    if derive:
        return logistic(z) * (1 - logistic(z))
    else:
        return 1 / (1 + np.exp(-z))

def sigmoid(z, derive=False):
    # from LeCun
    fun          = np.tanh(2/3 * z)
    derivative_1 = 2/3 * 1/np.cosh(z)**2 # float div is python3 only
    return derivative_1 if derive else fun

def network_fun(x, weights):
    """Compute the network output for given data and weights."""
    return sigmoid(weights[1] * sigmoid(weights[0] * x))

def generate_data(sampleSize=30):
    cats = np.random.normal(25, 5, sampleSize)
    dogs = np.random.normal(45, 15, sampleSize)
    data = np.concatenate((cats,dogs))
    # target_values from -1 to 1
    targets = np.concatenate((np.full(sampleSize, -1, dtype=int), np.ones(sampleSize)))
    return data, targets

def pre_process_data(data):
    """Mean-shift and variance-normalise the data."""
    data -= np.mean(data, 0)
    data /= np.std(data) # == sqrt(cov(data))
    return data # not sure if data is copied

def forward_pass(x, weights):
    o0 = sigmoid(weights[0] * x)
    o1 = sigmoid(weights[1] * o0)
    return o0, o1

def backpropagate(x, o0, o1, weights, t, learning_rate):
    """Do one step of backpropagation."""
    # from Mitchell
    delta1 = (t - o1) * sigmoid(o0 * weights[1], derive=True)
    delta0 = sigmoid(x*weights[0], derive = True) * weights[0] * delta1
    delta_w1 = learning_rate * delta1 * o0
    delta_w0 = learning_rate * delta0 * x
    return delta_w0, delta_w1

def stochastic_gradient_descent(data, targets, learning_rate=.1, iterations=1000):
    """Train the mininet on data."""
    # this variant should go with the sigmoid from LeCun
    weights = np.random.normal(loc=0, scale=1/np.sqrt(2), size=2)
    n = len(targets)
    for k in range(iterations):
        index = np.random.randint(low=0, high=n)
        o0, o1 = forward_pass(data[index], weights)
        weights += backpropagate(data[index], o0, o1, weights, targets[index], learning_rate)
    return weights

def classify(x, weights):
    """Run an example through the net."""
    return network_fun(x, weights)

def calc_error(weights, data, targets):
    """Calculate the cumulative error function over all given examples."""
    return 0.5 * sum((targets - classify(data, weights))**2)

def calc_misclassification(weights, data, targets):
    """Compute the percentage of misclassified examples. In this case, a network
    output >= 0 is consdired a '1' label, an output < 0 a '0' label."""
    output = network_fun(data, weights)
    return np.count_nonzero(np.abs(data - output) > 1) / len(data)

def print_data(index):
    print("data[{}]: target = {} \nnetwork says: {}".format(index,
        targets[index], classify(data[index], trained_weights)))

if __name__ == "__main__":
    data, targets = generate_data()
    data = pre_process_data(data)
    trained_weights = stochastic_gradient_descent(data, targets, iterations=1000)
    print("%% misclassified on training set: %f" % calc_misclassification(trained_weights, data, targets))

    # plotting
    weights0 = np.arange(-10, 10, .1)
    weights1 = np.arange(-10, 10, .1)
    X, Y = np.meshgrid(weights0, weights1)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = calc_error(np.array([X[i,j], Y[i,j]]), data, targets)


    plt.figure()
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.contour(X, Y, Z, colors="black", linestyles="dashed")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cp = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
    plt.colorbar(cp)
    plt.show()
