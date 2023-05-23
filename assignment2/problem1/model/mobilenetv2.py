import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_factor = expansion_factor
        self.stride = stride
        hidden_channels = in_channels * expansion_factor
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=stride, groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        
        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        
    def forward(self, x):
        out = self.layer(x)
        if self.stride == 1:
            out = (out + self.shortcut(x))
        return out            
        
class MobileNetV2(nn.Module):
    def __init__(self, t=6, num_class=100):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.layer = nn.Sequential(
            self.make_layer(32, 16, 1, 1, 1),
            self.make_layer(16, 24, t, 2, 2),
            self.make_layer(24, 32, t, 2, 3),
            self.make_layer(32, 64, t, 2, 4),
            self.make_layer(64, 96, t, 1, 3),
            self.make_layer(96, 160, t, 2, 3),
            self.make_layer(160, 320, t, 1, 1),
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),        
        )
        
        self.last_conv = nn.Conv2d(1280, num_class, kernel_size=1)
        
    
    def make_layer(self, in_channels, out_channels, expansion_factor, stride, block_num):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, expansion_factor, stride))
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, expansion_factor, 1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.pre(x)
        x = self.layer(x)
        x = self.last_conv(x)
        x = x.view(x.shape[0], -1)
        return x
    
        