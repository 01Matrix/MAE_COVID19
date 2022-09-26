# encoding=utf-8
from cProfile import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

acc_orig = [
0.705882353,
0.705882353,
0.633689840,
0.729946524,
0.727272727
]

acc_sani = [
0.675862069,
0.710344828,
0.675862069,
0.689655172,
0.696551724
]

# acc_orig_attn = [
# 0.705882353] 
# acc_sani_attn = [
# 0.710344828]

# acc_orig_mlp = [
# ] 
# acc_sani_mlp = [
# 0.675862069]
# acc_orig_norm1=[
# 0.729946524]
# acc_sani_norm1=[
# 0.689655172]
# acc_orig_norm2=[
# 0.727272727]
# acc_sani_norm2=[
# 0.696551724]

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
plt.savefig('./barchart_CXCbase_sublastblock.png')
# plt.close()

