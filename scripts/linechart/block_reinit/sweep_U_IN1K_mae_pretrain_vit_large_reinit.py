# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
x = range(len(names))

acc_orig = [
0.855614973,
0.839572193,
0.823529412,
0.826203209,
0.818181818,
0.77540107,
0.826203209,
0.735294118,
0.799465241,
0.689839572,
0.786096257,
0.77540107,
0.71657754,
0.754010695,
0.810160428,
0.79144385,
0.815508021,
0.745989305,
0.778074866,
0.794117647,
0.735294118,
0.695187166,
0.684491979,
0.647058824,
0.532085561,
] 
acc_sani = [
0.744827586,
0.765517241,
0.751724138,
0.75862069,
0.786206897,
0.696551724,
0.75862069,
0.820689655,
0.751724138,
0.696551724,
0.75862069,
0.779310345,
0.793103448,
0.786206897,
0.703448276,
0.786206897,
0.772413793,
0.744827586,
0.668965517,
0.751724138,
0.75862069,
0.634482759,
0.737931034,
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
plt.savefig('../savedfig/linechart_blocks_acc_maelarge_reinit.png')
plt.close()
plt.clf()
