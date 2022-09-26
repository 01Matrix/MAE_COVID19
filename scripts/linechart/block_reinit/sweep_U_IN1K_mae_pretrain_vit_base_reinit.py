# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
x = range(len(names))

acc_orig = [
0.847593583,
0.836898396,
0.794117647,
0.847593583,
0.860962567,
0.826203209,
0.764705882,
0.767379679,
0.756684492,
0.614973262,
0.657754011,
0.644385027,
0.532085561,
] 
acc_sani = [
0.772413793,
0.820689655,
0.827586207,
0.793103448,
0.779310345,
0.779310345,
0.731034483,
0.772413793,
0.751724138,
0.751724138,
0.710344828,
0.606896552,
0.544827586,
]



#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
# plt.xlim(0, 5) # 限定横轴的范围
# plt.ylim(0, 1) # 限定纵轴的范围

# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
# plt.rcParams['figure.figsize'] = (80,40)
# plt.rcParams['axes.titlesize'] = 15 #子图的标题大小
# plt.rcParams['axes.labelsize'] = 15 #子图的标签大小
# plt.rcParams['xtick.labelsize'] = 15 
# plt.rcParams['ytick.labelsize'] = 15 


plt.plot(x, acc_orig, linewidth=1.0, label='U_orig')
plt.plot(x, acc_sani, linewidth=1.0, label='U_sani')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of re-initialized higher MAE encoder blocks") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base") #标题
plt.savefig('../savedfig/linechart_blocks_acc_maebase_reinit.png')
plt.close()
plt.clf()
