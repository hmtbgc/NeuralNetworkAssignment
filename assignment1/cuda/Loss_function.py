import cupy as cp

class CrossEntropyLoss():
    
    def __init__(self):
        pass
    
    # forward
    # X shape: (n, k), not pass through softmax layer
    # label shape: (n, )
    # loss = -sum(y * log(y_output)) / n 
    
    def forward(self, X, label):
        self.label = label
        X_max = cp.max(X, axis=1, keepdims=True)
        self.X = X - X_max
        exp_sum = cp.sum(cp.exp(self.X), axis=1, keepdims=True)
        self.X = cp.exp(self.X) / exp_sum
        self.n = self.X.shape[0]
        loss = -cp.sum(cp.log(self.X[cp.arange(self.n), label])) / self.n
        return loss

    # backward
    
    def backward(self):
        label_onehot = cp.zeros_like(self.X)
        label_onehot[cp.arange(self.n), self.label] = 1.0
        Current_Gradient = (self.X - label_onehot) / self.n
        return Current_Gradient
