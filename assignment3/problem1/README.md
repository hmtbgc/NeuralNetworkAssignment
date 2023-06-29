### 问题概述
- 任务: 图像分类
- 数据集: CIFAR-10
- CNN Backbone: ResNet18
- 方法：self-supervised + linear evaluation 和 supervised，其中self-supervised 使用的算法是 SimCLR

Top1 Accuracy
| SimCLR+linear evalutation  | supervised|
|  ----  | ---- |
| 81.05%| 93.68%|