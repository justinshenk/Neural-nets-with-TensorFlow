import struct
import numpy as np


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
