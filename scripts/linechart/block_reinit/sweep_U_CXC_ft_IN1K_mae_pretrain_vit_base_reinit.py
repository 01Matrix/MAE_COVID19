# encoding=utf-8
from ssl import ALERT_DESCRIPTION_ACCESS_DENIED
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
x = range(len(names))

acc_orig =[
0.877005348,
0.860962567,
0.847593583,
0.834224599,
0.847593583,
0.842245989,
0.85828877,
0.767379679,
0.729946524,
0.655080214,
0.689839572,
0.647058824,
0.532085561,
]

acc_sani=[
0.827586207,
0.820689655,
0.848275862,
0.868965517,
0.868965517,
0.862068966,
0.75862069,
0.751724138,
0.786206897,
0.731034483,
0.717241379,
0.710344828,
0.579310345,    
]



plt.plot(x, acc_orig, linewidth=1.0, label='U_orig')
plt.plot(x, acc_sani, linewidth=1.0, label='U_sani')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of re-initialized higher MAE encoder blocks") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base") #标题
plt.savefig('../savedfig/linechart_blocks_acc_cbase_reinit.png')
plt.close()
plt.clf()
