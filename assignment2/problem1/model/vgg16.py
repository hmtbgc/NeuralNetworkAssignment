import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, num_class=100):
        super().__init__()        
        self.layer1 = nn.Sequential(
            self.make_layer(3, 64),
            self.make_layer(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_layer(64, 128),
            self.make_layer(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_layer(128, 256),
            self.make_layer(256, 256),
            self.make_layer(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_layer(256, 512),
            self.make_layer(512, 512),
            self.make_layer(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_layer(512, 512),
            self.make_layer(512, 512),
            self.make_layer(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_class),
        )
        
    def make_layer(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.shape[0], -1)
        return self.layer2(x)