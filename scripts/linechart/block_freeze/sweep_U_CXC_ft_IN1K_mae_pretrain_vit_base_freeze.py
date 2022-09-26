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
0.877005348,
0.868983957,
0.855614973,
0.863636364,
0.871657754,
0.85026738,
0.834224599,
0.863636364,
0.812834225,
0.743315508,
0.762032086,
0.705882353,
0.727272727,
] 
acc_sani = [
0.827586207,
0.84137931,
0.84137931,
0.855172414,
0.84137931,
0.875862069,
0.848275862,
0.875862069,
0.848275862,
0.786206897,
0.731034483,
0.675862069,
0.689655172,
]


plt.plot(x, acc_orig, linewidth=1.0, label='U_orig')
plt.plot(x, acc_sani, linewidth=1.0, label='U_sani')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of frozen lower MAE encoder blocks") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base") #标题
plt.savefig('../savedfig/linechart_blocks_acc_cbase_freeze.png')
plt.close()
plt.clf()