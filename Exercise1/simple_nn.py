import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z, derive=False):
    # from LeCun
    fun = 1.7159 * np.tanh(2/3 * z)
    derivative_1 = 1.7159 * 2/3 * 1/np.cosh(z)**2 # float div is python3 only
    return derivative_1 if derive else fun

def generate_data():
    sampleSize = 30
    np.random.seed(1)
    cats = np.random.normal(25, 5, sampleSize)
    dogs = np.random.normal(45, 15, sampleSize)
    

