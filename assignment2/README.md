### problem1
- Task: image classification
- Dataset: CIFAR-100
- CNN Architecture: VGG16, ResNet18, MobileNetV2
- Data Augmentation: Mixup, Cutout, Cutmix

Top1 error
|  Method  | Baseline  | Mixup($\alpha=0.4$)|Cutout|Cutmix($\alpha=0.4$)|
|  ----  | ----  |----|----|----|
| VGG16  | 29.32% |27.65%|29.49%|26.70%|
| ResNet18  |24.39% |22.73%|24.72%|22.12%|
|MobileNetV2|33.67%|32.47%|34.57%|36.07%|

Top5 error
|  Method  | Baseline  |Mixup ($\alpha=0.4$)|Cutout|Cutmix($\alpha=0.4$)|
|  ----  | ----  |----|----|----|
| VGG16  | 11.09% |9.36%|11.25%|8.46%|
| ResNet18  |7.47% |6.91%|7.23%|5.89%|
|MobileNetV2|10.11%|9.64%|10.12%|11.43%|

#### Ablation study

1. Mixup $\alpha$, Top1 error

|  Method  | Mixup ($\alpha=0.2$) |Mixup ($\alpha=0.4$)|Mixup ($\alpha=0.6$)|
|  ----  | ----  |----|----|
| VGG16  | 27.66% |27.65%|28.00%|
| ResNet18  | 23.86%|22.73%|23.76%|
|MobileNetV2|33.09%|32.47%|33.82%|

2. Cutmix $\alpha$, Top1 error

|  Method  | Cutmix($\alpha=0.2$) |Cutmix($\alpha=0.4$)|Cutmix($\alpha=0.6$)|
|  ----  | ----  |----|----|
| VGG16  |26.82%  |26.70%|26.57%|
| ResNet18  |22.60% |22.12%|22.19%|
|MobileNetV2|35.86%|36.07%|36.34%|

#### Visualization
raw picture

![](./problem1/test_pic/bird_re.jpg)
![](./problem1/test_pic/cat_re.jpg)
![](./problem1/test_pic/dog_re.jpg)

cutout result

![](./problem1/visualization/bird_cutout.png)
![](./problem1/visualization/cat_cutout.png)
![](./problem1/visualization/dog_cutout.png)

mixup result

![](./problem1/visualization/bird_mixup_cat.png)
![](./problem1/visualization/cat_mixup_dog.png)
![](./problem1/visualization/dog_mixup_bird.png)

cutmix result

![](./problem1/visualization/bird_cutmix_cat.png)
![](./problem1/visualization/cat_cutmix_dog.png)
![](./problem1/visualization/dog_cutmix_bird.png)

