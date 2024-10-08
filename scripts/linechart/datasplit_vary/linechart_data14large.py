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
# lineplot_sweep_U_data14_mae_pretrain_vit_large_datasplit
acc_sani = [
0.544827586,
0.544827586,
0.662068966,
0.544827586,
0.544827586,
0.572413793,
0.551724138,
0.627586207,
0.675862069,
0.586206897,
0.544827586,
0.655172414,
0.765517241,
0.662068966,
0.634482759,
0.544827586,
0.689655172,
0.696551724,
0.544827586,
0.544827586,
0.675862069,
0.544827586,
0.544827586,
0.524137931,
0.55862069,
0.737931034,
0.744827586,
0.634482759,
0.634482759,
0.77241379,
0.682758621,
0.682758621,
0.703448276,
0.717241379,
0.675862069,
0.724137931,
0.737931034,
0.662068966,
0.662068966,
0.655172414,
0.772413793,
0.655172414,
0.682758621,
0.703448276,
0.703448276,
0.627586207
]
f1_sani = [
0,
0,
0.65248227,
0,
0,
0.569444444,
0.557823129,
0.557377049,
0.675862069,
0.189189189,
0,
0.444444444,
0.734375,
0.67114094,
0.686390533,
0,
0.720496894,
0.698630137,
0,
0,
0.711656442,
0,
0,
0.642487047,
0.573333333,
0.743243243,
0.744827586,
0.662420382,
0.686390533,
0.74015748,
0.7125,
0.646153846,
0.715231788,
0.751515152,
0.711656442,
0.726027397,
0.736111111,
0.642335766,
0.675496689,
0.666666667,
0.722689076,
0.662162162,
0.729411765,
0.736196319,
0.681481481,
0.696629213
]
auc_sani = [
0.515247411,
0.334675873,
0.735711546,
0.704545455,
0.519850403,
0.665228232,
0.654967395,
0.686133487,
0.745397008,
0.726409666,
0.774932873,
0.781070196,
0.810797852,
0.798139624,
0.798619102,
0.723532796,
0.799386268,
0.766781741,
0.777330265,
0.460490986,
0.815400844,
0.761219793,
0.80552359,
0.753835827,
0.670310702,
0.864307633,
0.846183353,
0.797851937,
0.806962025,
0.83477177,
0.823839662,
0.798714998,
0.756712697,
0.871308017,
0.816455696,
0.836881473,
0.827675489,
0.789029536,
0.795166858,
0.793536632,
0.834579977,
0.765439202,
0.855581128,
0.827100115,
0.800153433,
0.804852321
]

# ***************************
acc_orig = [
0.668449198,
0.537433155,
0.719251337,
0.684491979,
0.532085561,
0.665775401,
0.695187166,
0.537433155,
0.687165775,
0.612299465,
0.695187166,
0.72459893,
0.687165775,
0.705882353,
0.636363636,
0.63368984,
0.697860963,
0.601604278,
0.772727273,
0.689839572,
0.721925134,
0.609625668,
0.770053476,
0.799465241,
0.679144385,
0.831550802,
0.810160428,
0.644385027,
0.532085561,
0.532085561,
0.844919786,
0.852941176,
0.679144385,
0.703208556,
0.828877005,
0.532085561,
0.85026738,
0.852941176,
0.703208556,
0.834224599,
0.847593583,
0.85026738,
0.871657754,
0.890374332,
0.828877005,
0.713903743
]

f1_orig = [
0.583892617,
0.02259887,
0.690265487,
0.644578313,
0,
0.649859944,
0.645962733,
0.02259887,
0.674094708,
0.611260054,
0.7,
0.732467532,
0.67768595,
0.685714286,
0.609195402,
0.620498615,
0.712468193,
0.504983389,
0.785894207,
0.714285714,
0.688622754,
0.348214286,
0.779487179,
0.791086351,
0.605263158,
0.82739726,
0.812664908,
0.652741514,
0,
0,
0.844919786,
0.847645429,
0.627329193,
0.72319202,
0.835897436,
0,
0.854166667,
0.847645429,
0.690807799,
0.830601093,
0.842105263,
0.84,
0.866666667,
0.887671233,
0.832460733,
0.680597015
]

auc_orig = [
0.726145011,
0.746676238,
0.799569275,
0.744077531,
0.670495334,
0.773007897,
0.745297918,
0.665312276,
0.744651831,
0.671600861,
0.758521177,
0.811514716,
0.770480976,
0.770681981,
0.692031587,
0.693610912,
0.756008615,
0.680875808,
0.876582915,
0.771442929,
0.792648959,
0.756697775,
0.875233309,
0.891873654,
0.72183776,
0.831550802,
0.889404164,
0.754371859,
0.468126346,
0.733984207,
0.912720747,
0.922555635,
0.723129935,
0.797171572,
0.90080402,
0.722914573,
0.85026738,
0.925441493,
0.793237617,
0.89913855,
0.921076813,
0.922110553,
0.925987078,
0.946719311,
0.828877005,
0.713903743
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
# plt.title("Data14_mae_pretrain_vit_large + U_orig") #标题
# plt.savefig('../savedfig/Data14_mae_pretrain_vit_large_U_orig.png')
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
# plt.title("Data14_mae_pretrain_vit_large + U_sani") #标题
# # plt.show()
# plt.savefig('../savedfig/Data14_mae_pretrain_vit_large_U_sani.png')

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
# plt.title('data14large',fontsize=14)
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

plt.savefig('./linechart_data14large.png')