class L2():
    def __init__(self, lam):
        self.lam = lam

    def get_grads(self, model):
        params = model.get_params()
        return [2.0 * self.lam * param for param in params]
        
    