数据集：https://pan.baidu.com/s/1WsH1tpx3l4J-uiJcgSafrw 提取码：6fpf

模型参数：https://pan.baidu.com/s/1-RLdy_0_s1dBDbKFCteUHg 提取码：3jml

### 运行方式

```shell

tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

应该有如下目录结构：
```shell
fasterrcnn/VOCdevkit/                           
fasterrcnn/VOCdevkit/VOCcode/                   
fasterrcnn/VOCdevkit/VOC2007  
```

直接测试
```shell
# 解压上述模型参数，得到best_epoch_weights.pth，将其放置在logs/下

# 若想要得到第一阶段proposal box，将图片放在img/test1/下，后缀为.jpg

python get_proposal_box.py # 交互式多次处理，只需输入图片名称，结果保存在img_out/test1/下

# 若想得到最终检测的结果，将图片放在img/test2/下，后缀为.jpg

python predict.py # 交互式多次处理，只需输入图片名称，结果保存在img_out/test2/下
```

验证测试集上mAP

```shell
python get_map.py
```

训练
```shell
# 解压上述模型参数，得到voc_weights_resnet.pth，将其放置在model_data/下
# 修改frcnn.py 的第19行，取消注释，并注释第20行
python train.py # 训练超参可在train.py中修改
```

