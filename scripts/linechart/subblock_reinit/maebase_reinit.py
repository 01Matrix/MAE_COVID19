# encoding=utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
x = range(len(names))

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

acc_orig_norm = [
0.695187166,
0.719251337,
0.754010695,
0.711229947,
0.700534759,
0.673796791,
0.70855615,
0.681818182,
0.673796791,
0.681818182,
0.665775401,
0.614973262,
0.534759358,
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

acc_sani_norm = [
0.627586207,
0.6,
0.648275862,
0.696551724,
0.737931034,
0.696551724,
0.751724138,
0.703448276,
0.75862069,
0.544827586,
0.744827586,
0.675862069,
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
plt.savefig('./linechart_attn_mlp_norm_maebase_reinit_orig.png')
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
plt.savefig('./linechart_attn_mlp_norm_maebase_reinit_sani.png')
plt.close()
plt.clf()