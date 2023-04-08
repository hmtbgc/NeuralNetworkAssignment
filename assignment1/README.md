数据集已经下载在data中，无需额外下载。

训练和测试方法：如果有gpu，可以调用cuda文件夹下main.py或main.ipynb，后者可以看到已经运行的结果。如果没有gpu，可以调用numpy文件夹下main.py或main.ipynb。main.py和main.ipynb均包含训练、验证、保存模型、加载模型、测试等环节，可阅读源码后按需修改相应代码。

执行方法都为 
```shell
python main.py
```
(有时候可能需要安装一些package)