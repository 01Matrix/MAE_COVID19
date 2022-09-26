# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_base_datasplit
acc_orig_bias = [
0.681818182,
0.622994652,
0.601604278,
0.622994652,
0.598930481,
0.609625668,
0.663101604,
0.665775401,
0.668449198,
0.663101604,
0.604278075,
0.644385027,
0.652406417,
] 


acc_sani_bias = [
0.675862069,
0.696551724,
0.606896552,
0.613793103,
0.620689655,
0.544827586,
0.572413793,
0.586206897,
0.689655172,
0.627586207,
0.544827586,
0.579310345,
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
plt.savefig('../savedfig/barchart_bias_acc_cbase_reinit.png')
# plt.close()

