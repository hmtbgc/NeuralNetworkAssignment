import numpy as np
import gzip
from struct import unpack
import os

class Mnist():
    
    def __init__(self, root):
        self.root = root
        self.train_image_name = "train-images-idx3-ubyte.gz"
        self.train_label_name = "train-labels-idx1-ubyte.gz"
        self.test_image_name = "t10k-images-idx3-ubyte.gz"
        self.test_label_name = "t10k-labels-idx1-ubyte.gz"
        
    def read_image(self, path):
        with gzip.open(path, "rb") as f:
            magic, num, rows, cols = unpack(">4I", f.read(16))
            image = np.frombuffer(f.read(), dtype = np.uint8).reshape(num, rows, cols)
        return image
    
    def read_label(self, path):
        with gzip.open(path, "rb") as f:
            magic, num = unpack(">2I", f.read(8))
            label = np.frombuffer(f.read(), dtype = np.uint8)
        return label
    
    def normalize(self, image):
        image = image.astype(np.float32) / 255.0
        return image
    
    def load(self, normalized=True):
        train_image = self.read_image(os.path.join(self.root, self.train_image_name))
        train_label = self.read_label(os.path.join(self.root, self.train_label_name))
        test_image = self.read_image(os.path.join(self.root, self.test_image_name))
        test_label = self.read_label(os.path.join(self.root, self.test_label_name))
        if (normalized):
            train_image = self.normalize(train_image)
            test_image = self.normalize(test_image)
        return train_image, train_label, test_image, test_label
        
        
        
        