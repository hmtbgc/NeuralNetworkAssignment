from Dataset import Mnist
from Activate_function import ReLULayer, SoftmaxLayer, SigmoidLayer
from Model import FullyConnectedLayer
from Loss_function import CrossEntropyLoss
from Dataloader import DataLoader
from Optimizer import SGD
from Regularization import L2
import numpy as np
import os

input_dim = 784
hidden_dim = 128
output_dim = 10
batch_size = 30
Epoch = 30
lr = 0.1
lam = 0.01
train_rate = 0.8
valid_freq = 5
dataset_root = "./data/mnist"
dataset = Mnist(dataset_root)
train_image, train_label, valid_image, valid_label, test_image, test_label = dataset.load(rate=train_rate)
print(f"train image shape: {train_image.shape}")
print(f"train label shape: {train_label.shape}")
print(f"valid image shape: {valid_image.shape}")
print(f"valid label shape: {valid_label.shape}")
print(f"test image shape: {test_image.shape}")
print(f"test label shape: {test_label.shape}")

train_dataloader = DataLoader(data=train_image, label=train_label, batch_size=batch_size)
valid_dataloader = DataLoader(data=valid_image, label=valid_label, batch_size=batch_size)
test_dataloader = DataLoader(data=test_image, label=test_label, batch_size=batch_size)
    

def eval(dataloader, model):
    acc = 0
    tot_number = 0
    for data, label in dataloader.get_batch():
        data = data.reshape(-1, 28 * 28)
        tot_number += data.shape[0]
        model_forward_result = model.forward(data)
        prob = softmax.forward(model_forward_result)
        out = np.argmax(prob, axis=1)
        acc += np.sum(out == label)
    return acc / tot_number

        
class Model():
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = FullyConnectedLayer(input_dim, hidden_dim)
        self.layer2 = ReLULayer()
        self.layer3 = FullyConnectedLayer(hidden_dim, output_dim)
        self.layer1.init_parameter()
        self.layer3.init_parameter()
        
    def forward(self, inputs):
        x = self.layer1.forward(inputs)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        return x
    
    def backward(self, Upstream_Gradient):
        gradient = self.layer3.backward(Upstream_Gradient)
        gradient = self.layer2.backward(gradient)
        gradient = self.layer1.backward(gradient)
        return gradient
    
    def get_params(self):
        params1 = self.layer1.get_params()
        params3 = self.layer3.get_params()
        return params1 + params3
    
    def get_grads(self):
        grads1 = self.layer1.get_grads()
        grads3 = self.layer3.get_grads()
        return grads1 + grads3
    
    def save_model(self, name):
        self.layer1.save_model(name=f"{name}_layer1")
        self.layer3.save_model(name=f"{name}_layer3")
    
    def load_model(self, path):
        self.layer1.load_model(path=f"{path}_layer1.npz")
        self.layer3.load_model(path=f"{path}_layer3.npz")


model = Model(input_dim, hidden_dim, output_dim)
loss_function = CrossEntropyLoss()
optimizer = SGD(lr)
softmax = SoftmaxLayer()
regularization = L2(lam)
best_valid_acc = 0.0

for epoch in range(Epoch):
    batch_num = 0
    loss = 0.0
    for x_train, y_train in train_dataloader.get_batch():
        x_train = x_train.reshape(-1, 28 * 28)
        model_forward_result = model.forward(x_train)
        loss += loss_function.forward(model_forward_result, y_train)
        # print(f"epoch {epoch}, batch {batch_num}, loss: {loss:.4f}")
        batch_num += 1
        gradient = loss_function.backward()
        gradient = model.backward(gradient)
        # model.step(lr)
        params, grads = model.get_params(), model.get_grads()
        # for i in range(len(params)):
        #     print(f"params[{i}].shape: {params[i].shape}")
        #     print(f"grads[{i}].shape: {grads[i].shape}")
        regularization_grads = regularization.get_grads(model)
        for grad, regularization_grad in zip(grads, regularization_grads):
            grad += regularization_grad
        optimizer.step(params, grads)
    if (epoch > 0 and epoch % valid_freq == 0):
        valid_acc = eval(valid_dataloader, model)
        if (valid_acc > best_valid_acc):
            best_valid_acc = valid_acc
            model.save_model("./model_params/Model")
    print(f"epoch {epoch}, loss: {loss:.4f}, best_valid_acc: {best_valid_acc: .4f}")

print("testing...")
test_model = Model(0, 0, 0)
test_model.load_model("./model_params/Model")
test_acc = eval(test_dataloader, test_model)

print(f"test acc: {test_acc:.4f}")
    

        
    

        
