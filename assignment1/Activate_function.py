import numpy as np

class ReLULayer():
    
    """

        Y = max(0, X)
    
    """
    
    def __init__(self):
        pass
    
    # forward
    
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    
    # backward
    
    def backward(self, Upstream_Gradient):
        Current_Gradient = Upstream_Gradient
        Current_Gradient[Upstream_Gradient < 0] = 0
        return Current_Gradient
    
    
class SigmoidLayer():
    
    """
    
        Y = 1 / (1 + exp(-X))
    
    """
    
    def __init__(self):
        pass
    
    # forward 
    
    def sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))
    
    def forward(self, X):
        self.X = X
        return self.sigmoid(X)
    
    # backward
    
    def backward(self, Upstream_Gradient):
        Current_Gradient = Upstream_Gradient
        dYdX = self.sigmoid(self.X) * (1.0 - self.sigmoid(self.X))
        Current_Gradient = Current_Gradient * dYdX
        return Current_Gradient
    
        
        
        