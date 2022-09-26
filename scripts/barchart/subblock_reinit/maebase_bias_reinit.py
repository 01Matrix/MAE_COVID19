# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_base_datasplit
acc_orig_bias = [
0.695187166,
0.681818182,
0.56684492,
0.713903743,
0.636363636,
0.532085561,
0.465240642,
0.467914439,
0.465240642,
0.673796791,
0.612299465,
0.532085561,
0.617647059,
] 


acc_sani_bias = [
0.634482759,
0.724137931,
0.64137931,
0.634482759,
0.455172414,
0.544827586,
0.475862069,
0.544827586,
0.455172414,
0.537931034,
0.544827586,
0.434482759,
0.544827586,
]


size = 13
x = np.arange(size)
a = np.random.random(size)

b = np.random.random(size)

c = np.random.random(size)

total_width, n = 0.6, 2

width = total_width / n

x = x - (total_width - width) / 2                     
# plt.figure(figsize=(40,60))   
plt.bar(x, acc_orig_bias, color = "darkorange", width=width, label='U_orig/bias')
plt.bar(x+width, acc_sani_bias, color = "grey", width=width, label='U_sani/bias')
plt.legend(loc='best')
plt.xlabel("Number of re-initializing bias terms of higher MAE encoder blocks")
plt.ylabel("Accuracy")
plt.xticks(x,['0','1','2','3','4','5','6','7','8','9','10','11','12'])
plt.savefig('../savedfig/barchart_bias_acc_maebase_reinit.png')
# plt.close()

