# encoding=utf-8
from cProfile import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

acc_orig = [
0.665775401,
0.609625668,
0.625668449,
0.582887701,
0.582887701,
]

acc_sani = [
0.551724138,
0.531034483,
0.531034483,
0.531034483,
0.531034483,
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
plt.savefig('./barchart_CXClarge_sublastblock.png')
# plt.close()

