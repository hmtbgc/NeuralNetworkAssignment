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
|MobileNetV2|38.92%|41.24%|41.50%||

Top5 error
|  Method  | Baseline  |Mixup ($\alpha=0.4$)|Cutout|Cutmix($\alpha=0.4$)|
|  ----  | ----  |----|----|----|
| VGG16  | 11.09% |9.36%|11.25%|8.46%|
| ResNet18  |7.47% |6.91%|7.23%|5.89%|
|MobileNetV2|12.25%|14.56%|14.52%||

Ablation study

1. Mixup $\alpha$, Top1 error

|  Method  | Mixup ($\alpha=0.2$) |Mixup ($\alpha=0.4$)|Mixup ($\alpha=0.6$)|
|  ----  | ----  |----|----|
| VGG16  |  |27.65%||
| ResNet18  | |22.73%||
|MobileNetV2||41.24%||

2. Cutmix $\alpha$, Top1 error

|  Method  | Cutmix($\alpha=0.2$) |Cutmix($\alpha=0.4$)|Cutmix($\alpha=0.6$)|
|  ----  | ----  |----|----|
| VGG16  |  |26.70%||
| ResNet18  | |22.12%||
|MobileNetV2||||

