import struct
import numpy as np

class MNIST():
    def __init__(self, directory = "./"):
        self.testData = self._load(directory + "t10k-images-idx3-ubyte")
        self.testLabels = self._load(directory + "t10k-labels-idx1-ubyte", True)
        self.trainingData = self._load(directory + "train-images-idx3-ubyte")
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
    n_test            = m.testData.shape[0]
    data_test         = m.testData[:n_test//split,:,:]
    labels_test       = m.testLabels[n_test//split:]
    data_validation   = m.testData[n_test//split:,:,:]
    labels_validation = m.testLabels[n_test//split:]

    return data_train, labels_train, data_test, labels_test, data_validation, labels_validation
