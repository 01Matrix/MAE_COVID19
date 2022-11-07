# encoding=utf-8
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

names = ['0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9','2.0',
'2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8','2.9','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7',
'3.8','3.9','4.0','4.1','4.2','4.3','4.4','4.5','4.6','4.7','4.8','4.9'
]
x = range(len(names))
# lineplot_sweep_U_mae_pretrain_vit_base_datasplit
acc_sani = [
0.544827586,
0.689655172,
0.710344828,
0.75862069,
0.696551724,
0.64137931,
0.682758621,
0.731034483,
0.806896552,
0.765517241,
0.813793103,
0.772413793,
0.8,
0.75862069,
0.820689655,
0.8,
0.8,
0.751724138,
0.786206897,
0.793103448,
0.779310345,
0.772413793,
0.827586207,
0.717241379,
0.717241379,
0.75862069,
0.772413793,
0.75862069,
0.744827586,
0.682758621,
0.772413793,
0.765517241,
0.75862069,
0.779310345,
0.724137931,
0.703448276,
0.751724138,
0.724137931,
0.724137931,
0.765517241,
0.793103448,
0.820689655,
0.779310345,
0.710344828,
0.8,
0.8
]
f1_sani = [
0,
0.64,
0.676923077,
0.695652174,
0.568627451,
0.662337662,
0.488888889,
0.654867257,
0.777777778,
0.75,
0.802919708,
0.769230769,
0.8,
0.672897196,
0.811594203,
0.788321168,
0.778625954,
0.723076923,
0.763358779,
0.745762712,
0.737704918,
0.762589928,
0.796747967,
0.585858586,
0.751515152,
0.724409449,
0.769230769,
0.755244755,
0.633663366,
0.726190476,
0.77852349,
0.673076923,
0.755244755,
0.746031746,
0.75,
0.722580645,
0.763157895,
0.756097561,
0.736842105,
0.776315789,
0.782608696,
0.805970149,
0.783783784,
0.543478261,
0.805369128,
0.794326241
]
auc_sani = [
0.666474875,
0.756808592,
0.800249329,
0.831319524,
0.803893364,
0.773973916,
0.782125048,
0.824798619,
0.863924051,
0.843785961,
0.887322593,
0.857690832,
0.864883007,
0.890103567,
0.890391254,
0.872458765,
0.88981588,
0.844265439,
0.838895282,
0.874856157,
0.864691216,
0.867855773,
0.90774837,
0.8738972,
0.846950518,
0.854526275,
0.893747603,
0.847621787,
0.879171461,
0.83208669,
0.879267357,
0.894802455,
0.835922516,
0.852416571,
0.863636364,
0.829593402,
0.880801688,
0.888761028,
0.873321826,
0.874952052,
0.866417338,
0.898830073,
0.874472574,
0.8783084,
0.902474108,
0.891446107
]

# ***************************
acc_orig = [
0.754010695,
0.72459893,
0.78342246,
0.778074866,
0.794117647,
0.799465241,
0.823529412,
0.810160428,
0.807486631,
0.807486631,
0.764705882,
0.799465241,
0.767379679,
0.852941176,
0.807486631,
0.831550802,
0.810160428,
0.786096257,
0.834224599,
0.839572193,
0.85026738,
0.839572193,
0.855614973,
0.812834225,
0.855614973,
0.847593583,
0.823529412,
0.863636364,
0.804812834,
0.86631016,
0.855614973,
0.847593583,
0.874331551,
0.887700535,
0.879679144,
0.898395722,
0.877005348,
0.868983957,
0.901069519,
0.85828877,
0.860962567,
0.868983957,
0.887700535,
0.911764706,
0.898395722,
0.877005348
]

f1_orig = [
0.726190476,
0.628158845,
0.767908309,
0.718644068,
0.776811594,
0.788732394,
0.817679558,
0.80653951,
0.798882682,
0.772151899,
0.698630137,
0.777448071,
0.692579505,
0.833836858,
0.789473684,
0.810810811,
0.812664908,
0.791666667,
0.826815642,
0.811320755,
0.837209302,
0.810126582,
0.838323353,
0.769736842,
0.844827586,
0.82674772,
0.784313725,
0.847761194,
0.745644599,
0.857954545,
0.841176471,
0.848806366,
0.860534125,
0.873493976,
0.862385321,
0.888235294,
0.85625,
0.843450479,
0.891495601,
0.861618799,
0.860215054,
0.865013774,
0.875,
0.907563025,
0.895604396,
0.875675676
]

auc_orig = [
0.829461594,
0.871744436,
0.875405599,
0.887537688,
0.866805456,
0.884149318,
0.884206748,
0.88132089,
0.876826992,
0.89781766,
0.880100503,
0.895017947,
0.905297918,
0.916582915,
0.903661163,
0.903661163,
0.88011486,
0.872720747,
0.906360373,
0.903359655,
0.927150036,
0.941421393,
0.924048816,
0.933065327,
0.935606604,
0.937200287,
0.949174444,
0.949361091,
0.941349605,
0.942153625,
0.942268485,
0.937946877,
0.953539124,
0.955534817,
0.952232592,
0.956712132,
0.959152907,
0.961450108,
0.959282125,
0.952002872,
0.944364681,
0.953883704,
0.958578607,
0.970739411,
0.964838478,
0.955391242
]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
# plt.xlim(0, 5) # 限定横轴的范围
# plt.ylim(0, 1) # 限定纵轴的范围

# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
# plt.rcParams['figure.figsize'] = (80,40)
# plt.rcParams['axes.titlesize'] = 15 #子图的标题大小
# plt.rcParams['axes.labelsize'] = 15 #子图的标签大小
# plt.rcParams['xtick.labelsize'] = 15 
# plt.rcParams['ytick.labelsize'] = 15 


# plt.plot(x, acc_orig, linewidth=1.0, label='Accuracy on U_orig')
# plt.plot(x, f1_orig, linewidth=1.0, label='F1 on U_orig')
# plt.plot(x, auc_orig,linewidth=1.0, label='AUC on U_orig')
# plt.legend() # 让图例生效
# plt.xticks(x, names, rotation=90)
# plt.margins(0)
# plt.xlabel("Training set portion") #X轴标签
# plt.ylabel("Performance Score") #Y轴标签
# plt.title("IN1K_mae_pretrain_vit_large + U_orig") #标题
# plt.savefig('../savedfig/IN1K_mae_pretrain_vit_large_U_orig.png')
# plt.close()
# plt.clf()

# plt.plot(x, acc_sani, linewidth=1.0, label='Accuracy on U_sani')
# plt.plot(x, f1_sani, linewidth=1.0, label='F1 on U_sani')
# plt.plot(x, auc_sani,linewidth=1.0, label='AUC on U_sani')
# plt.legend() # 让图例生效
# plt.xticks(x, names, rotation=90)
# plt.margins(0)
# plt.xlabel("Training set portion") #X轴标签
# plt.ylabel("Performance Score") #Y轴标签
# plt.title("IN1K_mae_pretrain_vit_large + U_sani") #标题
# # plt.show()
# plt.savefig('../savedfig/IN1K_mae_pretrain_vit_large_U_sani.png')


from scipy.interpolate import make_interp_spline
def smooth_xy(lx, ly):
    """数据平滑处理
    :param lx: x轴数据,数组
    :param ly: y轴数据,数组
    :return: 平滑后的x、y轴数据,数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 10000)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]

x_values=[x/10 for x in range(4,50,1)]

xy_s1 = smooth_xy(x_values,acc_orig)
xy_s2 = smooth_xy(x_values,acc_sani)
plt.plot(xy_s1[0],xy_s1[1],label='U_orig')
plt.plot(xy_s2[0],xy_s2[1],label='U_sani')

# plt.plot(x_values,acc_orig,label='U_orig')
# plt.plot(x_values,acc_sani,label='U_sani')
plt.legend()
# plt.title('maelarge',fontsize=14)
plt.tick_params(axis='both',which='major',labelsize=12)
plt.xlabel("Training set portion",fontsize=12) #X轴标签
plt.ylabel("Accuracy",fontsize=12) #Y轴标签
 
xmajorLocator = MultipleLocator(0.5) 		# 将x轴主刻度设置为0.4的倍数
xmajorFormatter = FormatStrFormatter('%1.1f') # 设置x轴标签的格式
xminorLocator = MultipleLocator(0.2) 		# 将x轴次刻度设置为0.2的倍数
ymajorLocator = MultipleLocator(0.1) 	# 将y轴主刻度设置为0.1的倍数
ymajorFormatter = FormatStrFormatter('%1.2f') # 设置y轴标签的格式
yminorLocator = MultipleLocator(0.05)	# 将y轴次刻度设置为0.1的倍数

ax = plt.gca()
# 设置主刻度标签的位置，标签文本的格式
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
 
# 显示次刻度标签的位置
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

plt.savefig('./linechart_maelarge.png')