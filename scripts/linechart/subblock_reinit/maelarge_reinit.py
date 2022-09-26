# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
x = range(len(names))
# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_large_datasplit
acc_orig_attn = [
0.818181818,
0.815508021,
0.802139037,
0.826203209,
0.767379679,
0.796791444,
0.778074866,
0.676470588,
0.56684492,
0.673796791,
0.561497326,
0.534759358,
0.561497326,
0.636363636,
0.617647059,
0.532085561,
0.692513369,
0.668449198,
0.679144385,
0.668449198,
0.462566845,
0.652406417,
0.697860963,
0.660427807,
0.679144385,
] 

acc_orig_mlp = [
0.834224599,
0.796791444,
0.788770053,
0.711229947,
0.796791444,
0.786096257,
0.770053476,
0.762032086,
0.751336898,
0.735294118,
0.729946524,
0.735294118,
0.703208556,
0.660427807,
0.660427807,
0.687165775,
0.673796791,
0.529411765,
0.644385027,
0.647058824,
0.655080214,
0.467914439,
0.593582888,
0.532085561,
0.502673797,
] 

acc_orig_norm = [
0.743315508,
0.751336898,
0.705882353,
0.727272727,
0.721925134,
0.762032086,
0.751336898,
0.721925134,
0.689839572,
0.751336898,
0.721925134,
0.732620321,
0.71657754,
0.727272727,
0.588235294,
0.729946524,
0.64171123,
0.729946524,
0.697860963,
0.660427807,
0.684491979,
0.582887701,
0.657754011,
0.588235294,
0.532085561,
]

acc_sani_attn = [
0.731034483,
0.765517241,
0.827586207,
0.813793103,
0.779310345,
0.744827586,
0.75862069,
0.786206897,
0.551724138,
0.772413793,
0.6,
0.455172414,
0.655172414,
0.544827586,
0.606896552,
0.55862069,
0.627586207,
0.579310345,
0.724137931,
0.710344828,
0.731034483,
0.544827586,
0.6,
0.689655172,
0.627586207,
]

acc_sani_mlp = [
0.668965517,
0.737931034,
0.75862069,
0.793103448,
0.793103448,
0.772413793,
0.751724138,
0.8,
0.786206897,
0.744827586,
0.662068966,
0.696551724,
0.689655172,
0.662068966,
0.682758621,
0.627586207,
0.620689655,
0.662068966,
0.737931034,
0.620689655,
0.655172414,
0.696551724,
0.593103448,
0.44137931,
0.544827586,
]

acc_sani_norm = [
0.655172414,
0.731034483,
0.682758621,
0.613793103,
0.737931034,
0.731034483,
0.731034483,
0.793103448,
0.724137931,
0.696551724,
0.737931034,
0.751724138,
0.737931034,
0.724137931,
0.662068966,
0.731034483,
0.703448276,
0.613793103,
0.731034483,
0.689655172,
0.689655172,
0.662068966,
0.455172414,
0.710344828,
0.544827586,
]


plt.plot(x, acc_orig_attn, linewidth=1.0, label='U_orig/Attention')
plt.plot(x, acc_orig_mlp, linewidth=1.0, label='U_orig/MLP')
plt.plot(x, acc_orig_norm, linewidth=1.0, label='U_orig/LayerNorm')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of higher MAE encoder blocks for re-initializing parts") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_large") #标题
plt.savefig('./linechart_attn_mlp_norm_maelarge_reinit_orig.png')
plt.close()
plt.clf()

plt.plot(x, acc_sani_attn, linewidth=1.0, label='U_sani/Attention')
plt.plot(x, acc_sani_mlp, linewidth=1.0, label='U_sani/MLP')
plt.plot(x, acc_sani_norm, linewidth=1.0, label='U_sani/LayerNorm')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of higher MAE encoder blocks for re-initializing parts") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_large") #标题
plt.savefig('./linechart_attn_mlp_norm_maelarge_reinit_sani.png')
plt.close()
plt.clf()