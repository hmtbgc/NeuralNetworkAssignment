import numpy as np

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
        self.W = np.zeros(shape=(hidden_dim, output_dim), dtype=np.float32)
        if (bias):
            self.b = np.zeros(shape=(1, output_dim), dtype=np.float32)
    
    def init_parameter(self):
        self.W = np.random.normal(loc=0.0, scale=1.0, size=(self.hidden_dim, self.output_dim))
        
    # forward
    
    def forward(self, X):
        self.X = X
        output = np.matmul(X, self.W)
        if (self.bias):
            output = output + self.bias
        return output
    
    # backward
    # Upstream_Gradient means dL/dY, shape: (n, k)
    
    def backward(self, Upstream_Gradient):
        self.dW = np.matmul(self.X.T, Upstream_Gradient)
        if (self.bias):
            self.db = np.sum(Upstream_Gradient, axis=0)
        Current_Graident = np.matmul(Upstream_Gradient, self.W.T)
        return Current_Graident
    
    # step
    
    def step(self, lr):
        self.W -= lr * self.dW
        if (self.bias):
            self.b -= lr * self.db
        