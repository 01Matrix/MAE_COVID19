# encoding=utf-8
from cProfile import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

acc_orig = [
0.780748663,
0.780748663,
0.794117647,
0.759358289,
0.732620321,
]

acc_sani = [
0.689655172,
0.662068966,
0.675862069,
0.655172414,
0.655172414,
]


size = 5
x = np.arange(size)

total_width, n = 0.4, 2

width = total_width / n

x = x - (total_width - width) / 2                     
# plt.figure(figsize=(40,60))   
plt.bar(x, acc_orig, color = "darkorange", width=width,label='U_orig')
plt.bar(x+width, acc_sani, color = "grey", width=width,label='U_sani')
plt.legend(loc='best')
plt.xlabel("Fine-tuning parts of the last MAE encoder block")
plt.ylabel("Accuracy")
plt.xticks(x,['Whole block','attn','mlp','norm1','norm2'])
plt.savefig('./barchart_maelarge_sublastblock.png')
# plt.close()

