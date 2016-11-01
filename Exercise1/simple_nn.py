# coding=utf-8"

__author__    = "Rasmus Diederichsen"
__date__      = "2016-11-01"
__email__     = "rdiederichse@uos.de"

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)

def sigmoid(z, derive=False):
    # from LeCun
    fun          = 1.7159 * np.tanh(2/3 * z)
    derivative_1 = 1.7159 * 2/3 * 1/np.cosh(z)**2 # float div is python3 only
    return derivative_1 if derive else fun

def generate_data(sampleSize=30):
    cats = np.random.normal(25, 5, sampleSize)
    dogs = np.random.normal(45, 15, sampleSize)
    data = np.concatenate((cats,dogs))
    targets = np.concatenate((np.ones(sampleSize), np.zeros(sampleSize)))
    return data, targets

def pre_process_data(data):
    data -= np.mean(data, 0)
    data /= np.std(data) # == sqrt(cov(data))
    return data

if __name__ == "__main__":
    data, targets = generate_data()
    data = pre_process_data(data)
    weights = np.random.normal(loc=0, scale=1/np.sqrt(2), size=2)
