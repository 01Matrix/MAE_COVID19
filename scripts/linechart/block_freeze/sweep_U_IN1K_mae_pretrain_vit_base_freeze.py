# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
x = range(len(names))
# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_base_datasplit
acc_orig = [
0.847593583,
0.807486631,
0.836898396,
0.847593583,
0.818181818,
0.794117647,
0.834224599,
0.77540107,
0.815508021,
0.810160428,
0.679144385,
0.77540107,
0.622994652,
] 

acc_sani = [
0.772413793,
0.820689655,
0.779310345,
0.682758621,
0.655172414,
0.793103448,
0.724137931,
0.786206897,
0.724137931,
0.655172414,
0.751724138,
0.696551724,
0.565517241,
]


plt.plot(x, acc_orig, linewidth=1.0, label='U_orig')
plt.plot(x, acc_sani, linewidth=1.0, label='U_sani')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of frozen lower MAE encoder blocks") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base") #标题
plt.savefig('../savedfig/linechart_blocks_acc_maebase_freeze.png')
plt.close()
plt.clf()