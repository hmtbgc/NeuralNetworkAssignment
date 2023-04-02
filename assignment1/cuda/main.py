from Dataset import Mnist
from Activate_function import ReLULayer, SoftmaxLayer, SigmoidLayer
from Model import FullyConnectedLayer
from Loss_function import CrossEntropyLoss
from Dataloader import DataLoader
import cupy as cp

input_dim = 784
hidden_dim = 128
output_dim = 10
batch_size = 512
Epoch = 100
lr = 0.1
dataset_root = "../data/mnist"
dataset = Mnist(dataset_root)
train_image, train_label, test_image, test_label = dataset.load()
print(f"train image shape: {train_image.shape}")
print(f"train label shape: {train_label.shape}")
print(f"test image shape: {test_image.shape}")
print(f"test label shape: {test_label.shape}")

train_dataloader = DataLoader(data=train_image, label=train_label, batch_size=batch_size)
test_dataloader = DataLoader(data=test_image, label=test_label, batch_size=batch_size)
    
        
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
    
    def step(self, lr):
        self.layer3.step(lr)
        self.layer1.step(lr)


model = Model(input_dim, hidden_dim, output_dim)
loss_function = CrossEntropyLoss()
softmax = SoftmaxLayer()

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
        model.step(lr)
    print(f"epoch {epoch}, loss: {loss:.4f}")

acc = 0
for x_test, y_test in test_dataloader.get_batch():
    x_test = x_test.reshape(-1, 28 * 28)
    model_forward_result = model.forward(x_test)
    prob = softmax.forward(model_forward_result)
    out = cp.argmax(prob, axis=1)
    acc += cp.sum(out == y_test)

print(f"test acc: {acc / test_image.shape[0]:.4f}")
    

        
    

        
