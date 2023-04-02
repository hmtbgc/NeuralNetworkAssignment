class SGD():
    def __init__(self, lr):
        self.lr = lr
        
    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad