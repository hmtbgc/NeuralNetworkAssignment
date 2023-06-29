from resnet import ResNet18
from ViT import VisionTransformer
from utils import *
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, backbone, in_feat, num_class):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(in_feat, num_class)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
    
# backbone = ResNet18()

# net = Net(backbone=backbone, in_feat=512, num_class=100)
# print(count_parameters(net)) # 11,220,132

backbone = VisionTransformer()
net = Net(backbone=backbone, in_feat=512, num_class=100)
print(count_parameters(net)) # 11,916,772




