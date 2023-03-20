import cupy as cp

class ReLULayer():
    
    """

        Y = max(0, X)
    
    """
    
    def __init__(self):
        pass
    
    # forward
    
    def forward(self, X):
        self.X = X
        return cp.maximum(0, X)
    
    # backward
    
    def backward(self, Upstream_Gradient):
        Current_Gradient = Upstream_Gradient
        Current_Gradient[self.X < 0] = 0
        return Current_Gradient
    
    
class SigmoidLayer():
    
    """
    
        Y = 1 / (1 + exp(-X))
    
    """
    
    def __init__(self):
        pass
    
    # forward 
    
    def sigmoid(self, X):
        return 1.0 / (1.0 + cp.exp(-X))
    
    def forward(self, X):
        self.X = X
        return self.sigmoid(X)
    
    # backward
    
    def backward(self, Upstream_Gradient):
        Current_Gradient = Upstream_Gradient
        dYdX = self.sigmoid(self.X) * (1.0 - self.sigmoid(self.X))
        Current_Gradient = Current_Gradient * dYdX
        return Current_Gradient
    

class SoftmaxLayer():
    
    """
    
        Y_i = exp(X_i - max(X_k)) / sum_j(exp(X_j - max(X_k)))
    
    """
    
    def __init__(self):
        pass
    
    def forward(self, X):
        # Only forward will be implemented
        X_max = cp.max(X, axis=1, keepdims=True)
        X = X - X_max
        exp_sum = cp.sum(cp.exp(X), axis=1, keepdims=True)
        X = cp.exp(X) / exp_sum
        return X
    
    def backward(self, Upstream_Gradient):
        # use CrossEntropyLoss to backward instead
        pass
        
        