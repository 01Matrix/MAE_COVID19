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
0.901069519,
0.898395722,
0.879679144,
0.877005348,
0.871657754,
0.882352941,
0.877005348,
0.893048128,
0.874331551,
0.860962567,
0.871657754,
0.901069519,
0.890374332,
0.885026738,
0.874331551,
0.874331551,
0.86631016,
0.855614973,
0.871657754,
0.810160428,
0.786096257,
0.593582888,
0.697860963,
0.665775401,
0.582887701,
] 
acc_sani = [
0.868965517,
0.868965517,
0.862068966,
0.868965517,
0.84137931,
0.855172414,
0.875862069,
0.875862069,
0.868965517,
0.875862069,
0.820689655,
0.875862069,
0.834482759,
0.868965517,
0.868965517,
0.875862069,
0.834482759,
0.793103448,
0.793103448,
0.737931034,
0.765517241,
0.586206897,
0.579310345,
0.551724138,
0.524137931,
]


plt.plot(x, acc_orig, linewidth=1.0, label='U_orig')
plt.plot(x, acc_sani, linewidth=1.0, label='U_sani')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of frozen lower MAE encoder blocks") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base") #标题
plt.savefig('../savedfig/linechart_blocks_acc_clarge_freeze.png')
plt.close()
plt.clf()