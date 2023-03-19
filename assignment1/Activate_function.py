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
    
    def __sigmoid__(self, X):
        return 1.0 / (1.0 + np.exp(-X))
    
    def forward(self, X):
        self.X = X
        return self.__sigmoid__(X)
    
    # backward
    
    def backward(self, Upstream_Gradient):
        Current_Gradient = Upstream_Gradient
        dYdX = self.__sigmoid__(self.X) * (1.0 - self.__sigmoid__(self.X))
        Current_Gradient = Current_Gradient * dYdX
        return Current_Gradient
        