# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
x = range(len(names))
# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_base_datasplit
acc_orig = [
0.855614973,
0.812834225,
0.834224599,
0.826203209,
0.826203209,
0.807486631,
0.823529412,
0.810160428,
0.796791444,
0.794117647,
0.831550802,
0.855614973,
0.794117647,
0.799465241,
0.842245989,
0.868983957,
0.85026738,
0.836898396,
0.786096257,
0.820855615,
0.767379679,
0.834224599,
0.794117647,
0.780748663,
0.732620321,
] 

acc_sani = [
0.744827586,
0.703448276,
0.765517241,
0.744827586,
0.731034483,
0.772413793,
0.703448276,
0.779310345,
0.765517241,
0.779310345,
0.772413793,
0.772413793,
0.765517241,
0.731034483,
0.751724138,
0.765517241,
0.703448276,
0.703448276,
0.744827586,
0.682758621,
0.737931034,
0.813793103,
0.717241379,
0.689655172,
0.655172414,
]


plt.plot(x, acc_orig, linewidth=1.0, label='U_orig')
plt.plot(x, acc_sani, linewidth=1.0, label='U_sani')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of frozen lower MAE encoder blocks") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base") #标题
plt.savefig('./linechart_blocks_acc_maelarge_freeze.png')
plt.close()
plt.clf()