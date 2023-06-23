import matplotlib.pyplot as plt

mAP = [0.23,0.40,0.53,0.63,0.7,0.78,0.8,0.813,0.82, 0.825]
plt.plot(mAP, linewidth=5)

# 设置图片标题，并给坐标轴x,y分别加上标签
plt.title('mAP of 10 epochs', fontsize=24)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('mAP', fontsize=18)

# 设置刻度标记的大小
plt.tick_params(axis='both', labelsize=14)
plt.show()