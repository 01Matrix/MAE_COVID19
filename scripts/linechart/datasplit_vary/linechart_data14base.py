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
# lineplot_sweep_U_data14_mae_pretrain_vit_base_datasplit
acc_sani = [
0.648275862,
0.613793103,
0.689655172,
0.634482759,
0.675862069,
0.689655172,
0.579310345,
0.668965517,
0.662068966,
0.544827586,
0.655172414,
0.696551724,
0.710344828,
0.703448276,
0.772413793,
0.634482759,
0.737931034,
0.786206897,
0.737931034,
0.710344828,
0.75862069,
0.634482759,
0.820689655,
0.662068966,
0.634482759,
0.75862069,
0.6,
0.786206897,
0.655172414,
0.703448276,
0.668965517,
0.648275862,
0.662068966,
0.703448276,
0.737931034,
0.682758621,
0.668965517,
0.682758621,
0.634482759,
0.675862069,
0.813793103,
0.8,
0.820689655,
0.710344828,
0.703448276,
0.689655172
]
f1_sani = [
0.564102564,
0.282051282,
0.594594595,
0.649006623,
0.661870504,
0.579439252,
0.371134021,
0.606557377,
0.642335766,
0,
0.671052632,
0.576923077,
0.7,
0.681481481,
0.751879699,
0.653594771,
0.736111111,
0.783216783,
0.660714286,
0.686567164,
0.75177305,
0.674846626,
0.796875,
0.662068966,
0.662420382,
0.740740741,
0.677777778,
0.786206897,
0.675324675,
0.729559748,
0.52,
0.514285714,
0.679738562,
0.729559748,
0.759493671,
0.634920635,
0.68,
0.701298701,
0.662420382,
0.704402516,
0.816326531,
0.802721088,
0.803030303,
0.712328767,
0.695035461,
0.671532847
]
auc_sani = [
0.7168201,
0.747506713,
0.739739164,
0.717299578,
0.738780207,
0.694188723,
0.666283084,
0.727943997,
0.747219026,
0.773782125,
0.773494438,
0.773398542,
0.796221711,
0.759685462,
0.865362486,
0.746355965,
0.796221711,
0.872650556,
0.879555044,
0.813387035,
0.852512466,
0.776083621,
0.872746452,
0.765822785,
0.749232835,
0.860280015,
0.713655543,
0.877541235,
0.751534331,
0.813291139,
0.772823168,
0.714039125,
0.782220944,
0.784906022,
0.848293057,
0.753356348,
0.761411584,
0.790851554,
0.757767549,
0.774645186,
0.885788262,
0.898830073,
0.886651323,
0.799578059,
0.7831799,
0.79756425
]

# ***************************
acc_orig = [
0.462566845,
0.628342246,
0.676470588,
0.740641711,
0.679144385,
0.711229947,
0.748663102,
0.684491979,
0.732620321,
0.695187166,
0.713903743,
0.748663102,
0.794117647,
0.740641711,
0.794117647,
0.812834225,
0.778074866,
0.78342246,
0.818181818,
0.804812834,
0.842245989,
0.673796791,
0.860962567,
0.86631016,
0.828877005,
0.863636364,
0.842245989,
0.863636364,
0.877005348,
0.703208556,
0.834224599,
0.871657754,
0.85026738,
0.890374332,
0.882352941,
0.847593583,
0.86631016,
0.831550802,
0.893048128,
0.871657754,
0.852941176,
0.871657754,
0.871657754,
0.893048128,
0.911764706,
0.919786096
]

f1_orig = [
0.629834254,
0.637075718,
0.640949555,
0.715542522,
0.54887218,
0.622377622,
0.682432432,
0.619354839,
0.735449735,
0.598591549,
0.723514212,
0.723529412,
0.790190736,
0.758104738,
0.774193548,
0.782608696,
0.75942029,
0.725423729,
0.786163522,
0.784660767,
0.829971182,
0.587837838,
0.845238095,
0.855491329,
0.8,
0.852173913,
0.818461538,
0.855524079,
0.873626374,
0.628762542,
0.820809249,
0.855421687,
0.83908046,
0.876876877,
0.865853659,
0.830860534,
0.863387978,
0.837209302,
0.88372093,
0.868131868,
0.852546917,
0.856287425,
0.851851852,
0.885057471,
0.904347826,
0.913793103
]

auc_orig = [
0.385829146,
0.731428571,
0.759755922,
0.796984925,
0.782842785,
0.732548457,
0.82246949,
0.756870065,
0.817501795,
0.80298636,
0.760387653,
0.84384781,
0.872333094,
0.822167983,
0.88310122,
0.912246949,
0.86379038,
0.895836324,
0.897343862,
0.888025844,
0.915075377,
0.773338119,
0.928557071,
0.935807609,
0.930064609,
0.933251974,
0.935922469,
0.940143575,
0.93896626,
0.831529074,
0.914343144,
0.955850682,
0.934730797,
0.95689878,
0.96017229,
0.937631012,
0.946489591,
0.91741565,
0.960703518,
0.950811199,
0.93689878,
0.943058148,
0.94373295,
0.964206748,
0.965656856,
0.970653266
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
# plt.title("Data14_mae_pretrain_vit_base + U_orig") #标题
# plt.savefig('../savedfig/Data14_mae_pretrain_vit_base_U_orig.png')
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
# plt.title("Data14_mae_pretrain_vit_base + U_sani") #标题
# # plt.show()
# plt.savefig('../savedfig/Data14_mae_pretrain_vit_base_U_sani.png')

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
# plt.title('data14base',fontsize=14)
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

plt.savefig('./linechart_data14base.png')