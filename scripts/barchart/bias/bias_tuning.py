# encoding=utf-8
from cProfile import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

acc_orig = [
0.7353,
0.6791,
0.6979,
0.5829
]

acc_sani = [
0.6207,
0.6897,
0.731,
0.6
]


size = 4
x = np.arange(size)

total_width, n = 0.2, 2

width = total_width / n

x = x - (total_width - width) / 2                     
# plt.figure(figsize=(40,60))   
plt.bar(x, acc_orig, color = "darkorange", width=width,label='U_orig')
plt.bar(x+width, acc_sani, color = "grey", width=width,label='U_sani')
plt.legend(loc='best')
plt.xlabel("Fine-tuning only bias terms")
plt.ylabel("Accuracy")
plt.xticks(x,['maebase','maelarge','CXCbase','CXClarge'])
plt.savefig('./barchart_bias_tuning.png')
# plt.close()

