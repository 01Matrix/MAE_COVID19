# encoding=utf-8
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

names = ['0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9','2.0',
'2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8','2.9','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7',
'3.8','3.9','4.0','4.1','4.2','4.3','4.4','4.5','4.6','4.7','4.8','4.9']
x = range(len(names))
# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_large_datasplit
acc_sani = [
0.565517241,
0.703448276,
0.724137931,
0.751724138,
0.744827586,
0.744827586,
0.820689655,
0.834482759,
0.834482759,
0.827586207,
0.848275862,
0.544827586,
0.875862069,
0.848275862,
0.868965517,
0.868965517,
0.848275862,
0.875862069,
0.855172414,
0.848275862,
0.882758621,
0.806896552,
0.848275862,
0.827586207,
0.875862069,
0.896551724,
0.834482759,
0.875862069,
0.868965517,
0.875862069,
0.868965517,
0.862068966,
0.875862069,
0.889655172,
0.862068966,
0.889655172,
0.862068966,
0.882758621,
0.875862069,
0.834482759,
0.868965517,
0.882758621,
0.868965517,
0.862068966,
0.889655172,
0.889655172
]
f1_sani = [
0.63583815,
0.736196319,
0.736842105,
0.772151899,
0.770186335,
0.758169935,
0.814285714,
0.815384615,
0.823529412,
0.825174825,
0.84057971,
0.459016393,
0.867647059,
0.842857143,
0.863309353,
0.863309353,
0.84057971,
0.867647059,
0.846715328,
0.833333333,
0.874074074,
0.808219178,
0.845070423,
0.827586207,
0.869565217,
0.888888889,
0.828571429,
0.863636364,
0.857142857,
0.861538462,
0.861313869,
0.850746269,
0.861538462,
0.880597015,
0.852941176,
0.880597015,
0.850746269,
0.872180451,
0.865671642,
0.826086957,
0.857142857,
0.874074074,
0.859259259,
0.846153846,
0.878787879,
0.878787879
]
auc_sani = [
0.565688531,
0.696298427,
0.792769467,
0.830552359,
0.806674338,
0.834771768,
0.892021481,
0.908803222,
0.900747986,
0.894514768,
0.911008822,
0.587552743,
0.922995781,
0.922995781,
0.934982739,
0.927119294,
0.916954354,
0.929133103,
0.91925585,
0.921077867,
0.926735712,
0.900172612,
0.911296509,
0.922420407,
0.919351745,
0.922516302,
0.915228232,
0.926735712,
0.923954737,
0.926639816,
0.936229382,
0.927982355,
0.935558113,
0.9380514,
0.931818182,
0.936996548,
0.936708861,
0.935941695,
0.933736095,
0.912351362,
0.929804373,
0.955696203,
0.941887227,
0.903624856,
0.9380514,
0.959340238
]
#********************
acc_orig = [
0.711229947,
0.820855615,
0.85026738,
0.820855615,
0.842245989,
0.844919786,
0.863636364,
0.885026738,
0.863636364,
0.890374332,
0.842245989,
0.85828877,
0.868983957,
0.86631016,
0.874331551,
0.85828877,
0.898395722,
0.874331551,
0.863636364,
0.860962567,
0.882352941,
0.901069519,
0.877005348,
0.879679144,
0.879679144,
0.879679144,
0.882352941,
0.874331551,
0.877005348,
0.898395722,
0.882352941,
0.871657754,
0.901069519,
0.890374332,
0.901069519,
0.882352941,
0.893048128,
0.882352941,
0.885026738,
0.911764706,
0.903743316,
0.879679144,
0.903743316,
0.877005348,
0.895721925,
0.874331551,
]

f1_orig = [
0.744075829,
0.816438356,
0.84,
0.813370474,
0.836565097,
0.832369942,
0.852173913,
0.874635569,
0.857142857,
0.88252149,
0.843501326,
0.84548105,
0.861189802,
0.858757062,
0.860534125,
0.856368564,
0.886904762,
0.854489164,
0.849557522,
0.857923497,
0.87283237,
0.895184136,
0.867816092,
0.871794872,
0.870317003,
0.873239437,
0.877094972,
0.866855524,
0.87150838,
0.894444444,
0.875,
0.868852459,
0.89212828,
0.887671233,
0.890855457,
0.877777778,
0.880239521,
0.86746988,
0.876080692,
0.904899135,
0.892857143,
0.866468843,
0.898305085,
0.868571429,
0.882882883,
0.86908078
]
auc_orig = [
0.829160086,
0.884005743,
0.89661163,
0.885326633,
0.893826274,
0.910868629,
0.921105528,
0.924982053,
0.926202441,
0.934788227,
0.91776023,
0.925915291,
0.930437904,
0.936697775,
0.937788945,
0.927681263,
0.947781766,
0.947997128,
0.945068198,
0.938176597,
0.943517588,
0.951457286,
0.942857143,
0.949389806,
0.953065327,
0.949432879,
0.945642498,
0.946891601,
0.952333094,
0.959784637,
0.950480976,
0.946518306,
0.955262024,
0.964924623,
0.96218234,
0.959842067,
0.95850682,
0.96166547,
0.948959081,
0.966403446,
0.961751615,
0.951170136,
0.955534817,
0.949274946,
0.969935391,
0.949461594
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
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_large + U_orig") #标题
# plt.savefig('../savedfig/CXC_ft_IN1K_mae_pretrain_vit_large_U_orig.png')
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
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_large + U_sani") #标题
# # plt.show()
# plt.savefig('../savedfig/CXC_ft_IN1K_mae_pretrain_vit_large_U_sani.png')

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
# plt.title('CXClarge',fontsize=14)
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

plt.savefig('./linechart_CXClarge.png')