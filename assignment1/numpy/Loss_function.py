import numpy as np

class CrossEntropyLoss():
    
    def __init__(self):
        pass
    
    # forward
    # X shape: (n, k), not pass through softmax layer
    # label shape: (n, )
    # loss = -sum(y * log(y_output)) / n 
    
    def forward(self, X, label):
        self.label = label
        X_max = np.max(X, axis=1, keepdims=True)
        self.X = X - X_max
        exp_sum = np.sum(np.exp(self.X), axis=1, keepdims=True)
        self.X = np.exp(self.X) / exp_sum
        self.n = self.X.shape[0]
        loss = -np.sum(np.log(self.X[np.arange(self.n), label])) / self.n
        return loss

    # backward
    
    def backward(self):
        label_onehot = np.zeros_like(self.X)
        label_onehot[np.arange(self.n), self.label] = 1.0
        Current_Gradient = (self.X - label_onehot) / self.n
        return Current_Gradient
    
