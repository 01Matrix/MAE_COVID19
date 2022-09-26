# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_base_datasplit
acc_orig_attn = [
0.844919786,
0.855614973,
0.85828877,
0.844919786,
0.855614973,
0.831550802,
0.826203209,
0.804812834,
0.705882353,
0.663101604,
0.780748663,
0.684491979,
0.684491979,
] 

acc_orig_mlp = [
0.860962567,
0.847593583,
0.893048128,
0.855614973,
0.852941176,
0.831550802,
0.842245989,
0.799465241,
0.863636364,
0.764705882,
0.719251337,
0.695187166,
0.631016043,
] 

acc_sani_attn = [
0.834482759,
0.820689655,
0.855172414,
0.827586207,
0.786206897,
0.772413793,
0.772413793,
0.793103448,
0.724137931,
0.731034483,
0.682758621,
0.579310345,
0.510344828,
]

acc_sani_mlp = [
0.793103448,
0.8,
0.834482759,
0.868965517,
0.862068966,
0.848275862,
0.820689655,
0.834482759,
0.855172414,
0.820689655,
0.717241379,
0.627586207,
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
plt.bar(x, acc_orig_attn, color = "darkorange", width=width, label='U_orig/attn')
plt.bar(x+width, acc_orig_mlp, color = "grey", width=width, label='U_orig/mlp')
plt.legend(loc='best')
plt.xlabel("Number of re-initializing attention adnd MLP parts of higher MAE encoder blocks")
plt.ylabel("Accuracy")
plt.xticks(x,['0','1','2','3','4','5','6','7','8','9','10','11','12'])
plt.savefig('../savedfig/barchart_attn_mlp_acc_cbase_reinit.png')
# plt.close()

