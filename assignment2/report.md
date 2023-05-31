# Assignment2 实验报告

## 问题1：图像分类
### 数据集介绍
我们采用的是CIFAR-100数据集，该数据集包含100种类别，每种类别有600张大小为32×32的RGB图像。每张图片有粗粒度和细粒度的两种标签，前者有20类，后者有100类。在实验中，我们选取100类的标签。

### 数据划分
针对每类的600张图片，我们选取其中500张为训练集，其余100张为测试集（为了比较公平，我们选用torchvision.datasets.CIFAR100中的train-test划分方式）。同时，随机选取10%的训练集作为验证集，来选取更优参数的模型。

### 网络结构
我们选取三种网络结构来进行比较。

#### VGG16

定义Conv(in, out, k, s, p)表示卷积操作，输入输出通道分别为in，out，卷积核大小为k×k，步长为s, padding为p；MaxPool2d(k, s)表示最大池化操作，池化核为k×k，步长为s。

整体流程：
(3, 32, 32)RGB图像:
-   Conv(3, 64, 3, 1, 1), Conv(64, 64, 3, 1, 1) -> (64, 32, 32)
-   MaxPool2d(2, 2) -> (64, 16, 16)
- Conv(64, 128, 3, 1, 1), Conv(128, 128, 3, 1, 1) -> (128, 16, 16)
- MaxPool2d(2, 2) -> (128, 8, 8)
- Conv(128, 256, 3, 1, 1), Conv(256, 256, 3, 1, 1), Conv(256, 256, 3, 1, 1) -> (256, 8, 8)
- MaxPool2d(2, 2) -> (256, 4, 4)
- Conv(256, 512, 3, 1, 1), Conv(512, 512, 3, 1, 1), Conv(512, 512, 3, 1, 1) -> (512, 4, 4)
- MaxPool2d(2, 2) -> (512, 2, 2)
- Conv(512, 512, 3, 1, 1), Conv(512, 512, 3, 1, 1), Conv(512, 512, 3, 1, 1) -> (512, 2, 2)
- MaxPool2d(2, 2) -> (512, 1, 1)
- Squeeze -> (512, )
- Linear(512, 4096) -> (4096, )
- Linear(4096, 4096) -> (4096, )
- Linear(4096, 100) -> (100, )
- Softmax -> Output

上述过程省略了激活函数、dropout和batch normalization的操作，这些操作的具体位置可以参考源码（一般而言，卷积层后跟BatchNorm2d和ReLU，线性层后跟ReLU和dropout，最后一个线性层后无需跟激活函数和dropout）。

#### ResNet18
定义ResidualBlock(in) 表示一个残差块，包括下面这些层的顺序组合：

Conv(in, in, 3, 1, 1), BatchNorm2d(in), ReLU, Conv(in, in, 3, 1, 1), BatchNorm2d(in), ShortCut, ReLU

其中ShortCut对应于原始论文中的skip-connection。

整体流程：(3, 32, 32) RGB图像：
- Conv(3, 64, 3, 1, 1), BatchNorm2d(64), ReLU -> (64, 32, 32)
- ResidualBlock(64), ResidualBlock(64) -> (64, 32, 32)
- Conv(64, 128, 3, 2, 1) -> (128, 16, 16)
- ResidualBlock(128), ResidualBlock(128) -> (128, 16, 16)
- Conv(128, 256, 3, 2, 1) -> (256, 8, 8)
- ResidualBlock(256), ResidualBlock(256) -> (256, 8, 8)
- Conv(256, 512, 3, 2, 1) -> (512, 4, 4)
- ResidualBlock(512), ResidualBlock(512) -> (512, 4, 4)
- AvgPool -> (512, 1, 1)
- Squeeze -> (512, )
- Linear(512, 100) -> (100, )
- Softmax -> Output

#### MobileNetV2
定义Conv(in, out, k, s, p, groups)表示一个组卷积操作，InvertedResidual(in, out, expansion_rate, s)表示一个逆残差块，其中hidden=in * expansion_rate。该残差块包括以下层：

Conv(in, hidden, 1, 1, 0), BatchNorm2d(hidden), ReLU6, Conv(hidden, hidden, 3, s, 1, hidden), BatchNorm2d(hidden), ReLU6, Conv(hidden, out, 1, 1, 0), BatchNorm2d(out), ShortCut(Optional)

注意，最后的ShortCut只有当s == 1 and in == out 时才奏效。

整体流程：(3, 32, 32) RGB图像：
- Conv(3, 32, 1, 1, 0), BatchNorm2d(32), ReLU6 -> (32, 32, 32)
- InvertedResidual(32, 16, 6, 1) -> (16, 32, 32)
- InvertedResidual(16, 24, 6, 2) -> (24, 16, 16)
- InvertedResidual(24, 24, 6, 1) -> (24, 16, 16)
- InvertedResidual(24, 32, 6, 2) -> (32, 8, 8)
- InvertedResidual(32, 32, 6, 1) -> (32, 8, 8)
- InvertedResidual(32, 32, 6, 1) -> (32, 8, 8)
- InvertedResidual(32, 64, 6, 2) -> (64, 4, 4)
- InvertedResidual(64, 64, 6, 1) -> (64, 4, 4)
- InvertedResidual(64, 64, 6, 1) -> (64, 4, 4)
- InvertedResidual(64, 64, 6, 1) -> (64, 4, 4)
- InvertedResidual(64, 96, 6, 1) -> (96, 4, 4)
- InvertedResidual(96, 96, 6, 1) -> (96, 4, 4)
- InvertedResidual(96, 96, 6, 1) -> (96, 4, 4)
- InvertedResidual(96, 160, 6, 2) -> (160, 2, 2)
- InvertedResidual(160, 160, 6, 1) -> (160, 2, 2)
- InvertedResidual(160, 160, 6, 1) -> (160, 2, 2)
- InvertedResidual(160, 320, 6, 1) -> (320, 2, 2)
- Conv(320, 1280, 1, 1, 0), BatchNorm2d(1280), ReLU6 -> (1280, 2, 2)
- AvgPool -> (1280, 1, 1)
- Conv(1280, 100, 1, 1, 0) -> (100, 1, 1)
- Squeeze -> (100, )
- Softmax -> Output

### 超参设置




