# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_base_datasplit
acc_orig_attn = [
0.810160428,
0.810160428,
0.820855615,
0.818181818,
0.767379679,
0.799465241,
0.810160428,
0.778074866,
0.788770053,
0.729946524,
0.727272727,
0.631016043,
0.665775401,
] 

acc_orig_mlp = [
0.823529412,
0.799465241,
0.842245989,
0.818181818,
0.810160428,
0.770053476,
0.72459893,
0.780748663,
0.815508021,
0.695187166,
0.681818182,
0.657754011,
0.631016043,
] 

acc_sani_attn = [
0.731034483,
0.779310345,
0.793103448,
0.710344828,
0.703448276,
0.8,
0.689655172,
0.710344828,
0.75862069,
0.772413793,
0.75862069,
0.717241379,
0.648275862,
]

acc_sani_mlp = [
0.737931034,
0.675862069,
0.779310345,
0.75862069,
0.772413793,
0.751724138,
0.710344828,
0.786206897,
0.64137931,
0.717241379,
0.668965517,
0.572413793,
0.613793103,
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
plt.bar(x, acc_orig_attn, color = "darkorange", width=width, label='U_orig/attn')
plt.bar(x+width, acc_orig_mlp, color = "grey", width=width, label='U_orig/mlp')
plt.legend(loc='best')
plt.xlabel("Number of re-initializing attention adnd MLP parts of higher MAE encoder blocks")
plt.ylabel("Accuracy")
plt.xticks(x,['0','1','2','3','4','5','6','7','8','9','10','11','12'])
plt.savefig('../savedfig/barchart_attn_mlp_acc_maebase_reinit.png')
# plt.close()

