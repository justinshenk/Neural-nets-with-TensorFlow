# coding=utf-8

__author__    = "Rasmus Diederichsen"
__date__      = "2016-11-01"
__email__     = "rdiederichse@uos.de"

import numpy as np
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
    # fun          = 1.7159 * np.tanh(2/3 * z)
    # derivative_1 = 1.7159 * 2/3 * 1/np.cosh(z)**2 # float div is python3 only
    fun          = np.tanh(2/3 * z)
    derivative_1 = 2/3 * 1/np.cosh(z)**2 # float div is python3 only
    return derivative_1 if derive else fun

def network_fun(x, weights):
    #return logistic(weights[1] * logistic(weights[0] * x))
    # dont tell me why there has to be a minus in front, but otherwise
    # the targetvalues are swapped
    return -sigmoid(weights[1] * sigmoid(weights[0] * x))

def generate_data(sampleSize=30):
    cats = np.random.normal(25, 5, sampleSize)
    dogs = np.random.normal(45, 15, sampleSize)
    data = np.concatenate((cats,dogs))
    # target_values from -1 to 1
    targets = np.concatenate((np.full(sampleSize, -1, dtype=int),(np.ones(sampleSize))))
    #targets = np.concatenate((np.ones(sampleSize), np.zeros(sampleSize)))
    return data, targets

def pre_process_data(data):
    """Mean-shift and variance-normalise the data."""
    data -= np.mean(data, 0)
    data /= np.std(data) # == sqrt(cov(data))
    return data # not sure if data is copied

def forward_pass(x, weights):
    # o0 = logistic(weights[0] * x)
    # o1 = logistic(weights[1] * o0)
    # return o0, o1
    o0 = sigmoid(weights[0] * x)
    o1 = sigmoid(weights[1] * o0)
    return o0, o1

def backpropagate(x, o0, o1, weights, t, learning_rate):
    # o1 = output of last neuron
    # o0 = output from first to outputneuron
    # delta1 = o1 * (1 - o1) * (t - o1)
    # delta0 = o0 * (1 - o0) * weights[0] * delta1
    # delta_w0 = learning_rate * delta0 * x
    # delta_w1 = learning_rate * delta1 * o0
    # return np.array([delta_w0, delta_w1])


    # tried to follow Mitchell and replace the activation function, not sure if
    # correct
    # the minus in front was wrong
    # delta1 = -(t - o1) * sigmoid(o0 * weights[1], derive=True)
    delta1 = (t - o1) * sigmoid(o0 * weights[1], derive=True)
    delta0 = sigmoid(x*weights[0], derive = True) * weights[0] * delta1
    delta_w1 = -learning_rate * delta1 * o0
    delta_w0 = learning_rate * delta0 * x
    return delta_w0, delta_w1

def stochastic_gradient_descent(data, targets, learning_rate=.1):
    # this variant should go with the sigmoid from LeCun
    # weights = np.random.normal(loc=0, scale=1/np.sqrt(2), size=2)
    weights = np.random.uniform(low=-.05,high=.05, size=2)
    n = len(targets)
    for k in range(100000):
        index = np.random.randint(low=0, high=n)
        o0, o1 = forward_pass(data[index], weights)
        weights += backpropagate(data[index], o0, o1, weights, targets[index], learning_rate)
    return weights

def classify(x, weights):
    return network_fun(x, weights)

def printdata(index):
    #print("data[",index,"] = {}, target = {}\nnetwork says {}".format(data[index],
    #    targets[index], classify(data[index], trained_weights)))
    print("data[{}]: target = {} \nnetwork says: {}".format(index,
        targets[index], classify(data[index], trained_weights)))

if __name__ == "__main__":
    data, targets = generate_data()
    data = pre_process_data(data)
    trained_weights = stochastic_gradient_descent(data, targets)
    X = trained_weights[0]
    Y = trained_weights[1]
    print("X: ",X, " Y: ", Y)
    printdata(0)
    printdata(1)
    printdata(55)
    printdata(56)
