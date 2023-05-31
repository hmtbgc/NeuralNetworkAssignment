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
无特殊说明，以下超参为所有测试通用
- batch size: 训练验证测试都是128。对每个epoch，都需要打乱训练集和验证集；测试集无需打乱。
- learning rate: 学习率先从0开始线性warm up到0.1，该过程持续一个epoch，然后每训练50个epoch，学习率乘0.2。
- optimizer: SGD，动量momentum为0.9，weight decay为5e-4。
- epoch：共训练200个epoch。
- loss function: CrossEntropyLoss
- 评价指标：accuracy，top1-error(1 - accuracy)，top5-error (1 - ground truth出现在score前5的概率)
- 验证频率：每2个epoch就进行一次验证集测试，以accuracy为是否保存模型参数的标准。

### 数据增强方法
#### Mixup($\alpha$)
先从$\Beta(\alpha, \alpha)$分布中选取$\lambda$，然后混合两个样本：$\hat{x}=\lambda x_1+(1-\lambda)x_2$，损失函数也要混合：$l(\hat{x})=\lambda l(x_1)+(1-\lambda)l(x_2)$。验证和测试时无需混合样本。

#### Cutout
对输入图上随机位置挖去若干块边长为$k$的方形区域，并设置该区域的值为0。实验中，选取$k=8$，并且只挖去一块。验证和测试时无需修剪样本。

#### Cutmix($\alpha$)
先从$\Beta(\alpha, \alpha)$分布中选取$\lambda$，然后在图$x_1$上随机位置挖去一块大小为($\lfloor\sqrt{1-\lambda}w\rfloor, \lfloor\sqrt{1-\lambda}h\rfloor$)的矩形区域，并用另一张图片$x_2$对应区域的内容填充。记挖去区域大小与全图面积大小的比值为$\mu$，则混合后损失函数为$l=(1-\mu)l(x_1)+\mu l(x_2)$。验证和测试时无需混合样本。


### 训练过程
以backbone为ResNet的baseline为例，展示tensorboard结果：
![](./problem1/tensorboard_log_pic/train_loss.png)
![](./problem1/tensorboard_log_pic/valid_loss.png)
![](./problem1/tensorboard_log_pic/valid_acc.png)
![](./problem1/tensorboard_log_pic/top1-error.png)
![](./problem1/tensorboard_log_pic/top5-error.png)

其中损失函数骤降和accuracy骤升的原因是学习率下降(乘0.2)，避免了局部最优解。

其余tensorboard结果可以通过如下方式获取：

```shell
# 先安装tensorboard：pip install tensorboard

cd tensorboard_log
tensorboard --logdir="./baseline/ResNet18" --port 6070

# baseline 和 ResNet18 可以替换成该目录下其他文件夹名称

# 最后打开浏览器，输入：
localhost:6070
```

### 测试结果

#### 不同backbone和数据增强方法的比较
我们考虑四种情形：不加入数据增强(baseline)，Mixup(0.4)，Cutout，Cutmix(0.4)。实验结果如下：

Top1 error
|  Method  | Baseline  | Mixup($0.4$)|Cutout|Cutmix($0.4$)|
|  ----  | ----  |----|----|----|
| VGG16  | 29.32% |27.65%|29.49%|**26.70%**|
| ResNet18  |24.39% |22.73%|24.72%|**22.12%**|
|MobileNetV2|33.67%|**32.47%**|34.57%|36.07%|

Top5 error
|  Method  | Baseline  |Mixup ($0.4$)|Cutout|Cutmix($0.4$)|
|  ----  | ----  |----|----|----|
| VGG16  | 11.09% |9.36%|11.25%|**8.46%**|
| ResNet18  |7.47% |6.91%|7.23%|**5.89%**|
|MobileNetV2|10.11%|**9.64%**|10.12%|11.43%|

针对不同的backbone，ResNet18表现的最好。在top1-error上，VGG16显著优于MobileNetV2；而在top5-error上，两者表现相近。

针对不同的数据增强方法，Mixup和Cutmix都能超过baseline(除了Cutmix在MobileNetV2上表现不佳)，且CutMix普遍优于Mixup。而Cutout难以超越baseline，可能与挖去的方形区域大小有密切联系。

#### 消融实验，针对不同的$\alpha$

1. Mixup $\alpha$, Top1 error

|  Method  | Mixup ($\alpha=0.2$) |Mixup ($\alpha=0.4$)|Mixup ($\alpha=0.6$)|
|  ----  | ----  |----|----|
| VGG16  | 27.66% |**27.65%**|28.00%|
| ResNet18  | 23.86%|**22.73%**|23.76%|
|MobileNetV2|33.09%|**32.47%**|33.82%|

2. Cutmix $\alpha$, Top1 error

|  Method  | Cutmix($\alpha=0.2$) |Cutmix($\alpha=0.4$)|Cutmix($\alpha=0.6$)|
|  ----  | ----  |----|----|
| VGG16  |26.82%  |26.70%|**26.57%**|
| ResNet18  |22.60% |**22.12%**|22.19%|
|MobileNetV2|**35.86%**|36.07%|36.34%|

总的来说，$\alpha$的不同取值对两种数据增强算法的影响较小。

#### 可视化三种数据增强算法
raw picture

![](./problem1/test_pic/bird_re.jpg)![](./problem1/test_pic/cat_re.jpg)![](./problem1/test_pic/dog_re.jpg)

mixup result

![](./problem1/visualization/bird_mixup_cat.png)![](./problem1/visualization/cat_mixup_dog.png)![](./problem1/visualization/dog_mixup_bird.png)

cutout result

![](./problem1/visualization/bird_cutout.png)![](./problem1/visualization/cat_cutout.png)![](./problem1/visualization/dog_cutout.png)

cutmix result

![](./problem1/visualization/bird_cutmix_cat.png)![](./problem1/visualization/cat_cutmix_dog.png)![](./problem1/visualization/dog_cutmix_bird.png)





