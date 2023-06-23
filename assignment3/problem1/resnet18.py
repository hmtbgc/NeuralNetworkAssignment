import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        
    def forward(self, x):
        return F.relu(x + self.layer(x))
     
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.block2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=(2, 2), padding=1),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        
        self.block6 = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512, out_dim)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view(x.shape[0], -1)
        #x = self.fc(x)
        return x