# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
x = range(len(names))

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

acc_orig_norm = [
0.689839572,
0.737967914,
0.743315508,
0.764705882,
0.77540107,
0.700534759,
0.721925134,
0.692513369,
0.692513369,
0.564171123,
0.631016043,
0.639037433,
0.556149733,
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

acc_sani_norm = [
0.689655172,
0.668965517,
0.703448276,
0.655172414,
0.75862069,
0.579310345,
0.544827586,
0.710344828,
0.565517241,
0.544827586,
0.648275862,
0.620689655,
0.544827586,
]


plt.plot(x, acc_orig_attn, linewidth=1.0, label='U_orig/Attention')
plt.plot(x, acc_orig_mlp, linewidth=1.0, label='U_orig/MLP')
plt.plot(x, acc_orig_norm, linewidth=1.0, label='U_orig/LayerNorm')
# plt.plot(x, acc_sani_attn, linewidth=1.0, label='U_sani/attn')
# plt.plot(x, acc_sani_mlp, linewidth=1.0, label='U_sani/mlp')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of higher MAE encoder blocks for re-initializing parts") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base") #标题
plt.savefig('./linechart_attn_mlp_norm_CXCbase_reinit_orig.png')
plt.close()
plt.clf()

plt.plot(x, acc_sani_attn, linewidth=1.0, label='U_sani/Attention')
plt.plot(x, acc_sani_mlp, linewidth=1.0, label='U_sani/MLP')
plt.plot(x, acc_sani_norm, linewidth=1.0, label='U_sani/LayerNorm')
# plt.plot(x, acc_sani_attn, linewidth=1.0, label='U_sani/attn')
# plt.plot(x, acc_sani_mlp, linewidth=1.0, label='U_sani/mlp')
plt.legend() # 让图例生效
plt.xticks(x, names, rotation=0)
plt.margins(0)
plt.xlabel("Number of higher MAE encoder blocks for re-initializing parts") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base") #标题
plt.savefig('./linechart_attn_mlp_norm_CXCbase_reinit_sani.png')
plt.close()
plt.clf()