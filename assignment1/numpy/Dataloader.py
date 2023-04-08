import numpy as np

class DataLoader():
    
    def __init__(self, data, label, batch_size, shuffle=True):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.length = self.data.shape[0]
        if (shuffle):
            permutation = np.random.permutation(self.data.shape[0])
            self.data = self.data[permutation]
            self.label = self.label[permutation] 
    
    def __len__(self):
        return (self.length - 1) // self.batch_size + 1
    
    def get_batch(self):
        for i in range(len(self)):
            begin = i * self.batch_size
            end = min((i + 1) * self.batch_size, self.length)
            yield self.data[begin : end], self.label[begin : end]
                
    