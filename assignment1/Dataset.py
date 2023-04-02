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
    
    def load(self, rate, normalized=True):
        train_image = self.read_image(os.path.join(self.root, self.train_image_name))
        train_label = self.read_label(os.path.join(self.root, self.train_label_name))
        test_image = self.read_image(os.path.join(self.root, self.test_image_name))
        test_label = self.read_label(os.path.join(self.root, self.test_label_name))
        if (normalized):
            train_image = self.normalize(train_image)
            test_image = self.normalize(test_image)
        tot_idx = np.arange(train_image.shape[0])
        np.random.shuffle(tot_idx)
        train_number = int(rate * tot_idx.shape[0])
        train_image, valid_image = train_image[:train_number], train_image[train_number:]
        train_label, valid_label = train_label[:train_number], train_label[train_number:]
        return train_image, train_label, valid_image, valid_label, test_image, test_label

        
        
        