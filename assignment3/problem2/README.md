### 问题概述
- 任务: 图像分类
- 数据集: CIFAR-100
- 模型：ResNet18、ViT
- 数据增强方法：Mixup, Cutout, Cutmix

Top1 error
|  Method  | Parameter|Baseline  | Mixup($\alpha=0.4$)|Cutout|Cutmix($\alpha=0.4$)|
|  ----  |----| ----  |----|----|----|
| ResNet18| 11,220,132 | 26.01% |25.73%|26.99%|24.57%|
| ViT  |11,916,772|50.64% |48.70%|53.08%|43.67%|


Top5 error
|  Method  | Parameter|Baseline  | Mixup($\alpha=0.4$)|Cutout|Cutmix($\alpha=0.4$)|
|  ----  |----| ----  |----|----|----|
| ResNet18| 11,220,132 | 7.53% |7.83%|8.11%|6.55%|
| ViT  |11,916,772|24.63% |22.72%|25.98%|17.32%|