import cupy as cp
from matplotlib import pyplot as plt

class FullyConnectedLayer():
    
    """
    
        Y = XW + b
        X: input, shape: (n, m)
        W: parameter matrix, shape: (m, k)
        b: bias vector, shape: (1, k)
        Y: output, shape: (n, k)
        
    """
    
    def __init__(self, hidden_dim, output_dim, bias=True):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        self.W = cp.zeros(shape=(hidden_dim, output_dim), dtype=cp.float32)
        if (bias):
            self.b = cp.zeros(shape=(1, output_dim), dtype=cp.float32)
    
    def init_parameter(self):
        self.W = cp.random.normal(loc=0.0, scale=1.0, size=(self.hidden_dim, self.output_dim))
        
    # forward
    
    def forward(self, X):
        self.X = X
        output = cp.matmul(X, self.W)
        if (self.bias):
            output = output + self.bias
        return output
    
    # backward
    # Upstream_Gradient means dL/dY, shape: (n, k)
    
    def backward(self, Upstream_Gradient):
        self.dW = cp.matmul(self.X.T, Upstream_Gradient)
        if (self.bias):
            self.db = cp.sum(Upstream_Gradient, axis=0, keepdims=True)
        Current_Graident = cp.matmul(Upstream_Gradient, self.W.T)
        return Current_Graident
        
    def get_params(self):
        if (self.bias):
            return [self.W, self.b]
        return [self.W]
    
    def get_grads(self):
        if (self.bias):
            return [self.dW, self.db]
        return [self.dW]
    
    def save_model(self, name):
        if (self.bias):
            cp.savez(name, hid=self.hidden_dim, out=self.output_dim, W=self.W, b=self.b)
        else:
            cp.savez(name, hid=self.hidden_dim, out=self.output_dim, W=self.W)
            
    def load_model(self, path):
        loaded_model = cp.load(path)
        self.hidden_dim = loaded_model["hid"]
        self.output_dim = loaded_model["out"]
        self.W = loaded_model["W"]
        if (self.bias):
            self.b = loaded_model["b"]
        
         
            
        
        
