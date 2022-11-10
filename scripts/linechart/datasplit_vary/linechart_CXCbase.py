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
# sweep_U_CXC_ft_IN1K_mae_pretrain_vit_base_datasplit
acc_sani = [
0.689655172,
0.606896552,
0.689655172,
0.668965517,
0.703448276,
0.8,
0.779310345,
0.8,
0.779310345,
0.75862069,
0.834482759,
0.8,
0.806896552,
0.820689655,
0.855172414,
0.84137931,
0.855172414,
0.813793103,
0.855172414,
0.848275862,
0.84137931,
0.84137931,
0.820689655,
0.862068966,
0.84137931,
0.848275862,
0.834482759,
0.848275862,
0.848275862,
0.834482759,
0.834482759,
0.834482759,
0.84137931,
0.84137931,
0.848275862,
0.862068966,
0.820689655,
0.848275862,
0.84137931,
0.834482759,
0.827586207,
0.855172414,
0.862068966,
0.834482759,
0.848275862,
0.896551724
]
f1_sani = [
0.733727811,
0.666666667,
0.697986577,
0.7,
0.729559748,
0.797202797,
0.786666667,
0.797202797,
0.780821918,
0.75177305,
0.815384615,
0.775193798,
0.787878788,
0.805970149,
0.842105263,
0.832116788,
0.848920863,
0.780487805,
0.842105263,
0.828125,
0.82962963,
0.818897638,
0.805970149,
0.84375,
0.81300813,
0.825396825,
0.823529412,
0.830769231,
0.828125,
0.820895522,
0.80952381,
0.80952381,
0.82962963,
0.836879433,
0.833333333,
0.850746269,
0.796875,
0.842857143,
0.827067669,
0.80952381,
0.817518248,
0.837209302,
0.848484848,
0.818181818,
0.828125,
0.888888889
]
auc_sani = [
0.767644802,
0.727560414,
0.79181051,
0.802263138,
0.818181818,
0.853471423,
0.84225163,
0.881281166,
0.874088991,
0.847238205,
0.888377445,
0.904679708,
0.890199463,
0.903241273,
0.90199463,
0.899980821,
0.92213272,
0.914556962,
0.921749137,
0.907556578,
0.921749137,
0.908036057,
0.902090526,
0.93872267,
0.927982355,
0.922036824,
0.912543153,
0.902857691,
0.920406598,
0.889336402,
0.90640583,
0.898062908,
0.893939394,
0.91925585,
0.910241657,
0.891733794,
0.908803222,
0.924817798,
0.909474492,
0.916474875,
0.920981972,
0.936708861,
0.938243191,
0.903337169,
0.936900652,
0.946202532
]

acc_orig = [
0.756684492,
0.807486631,
0.836898396,
0.823529412,
0.812834225,
0.842245989,
0.844919786,
0.842245989,
0.85026738,
0.847593583,
0.834224599,
0.860962567,
0.836898396,
0.860962567,
0.847593583,
0.871657754,
0.860962567,
0.868983957,
0.836898396,
0.847593583,
0.85026738,
0.885026738,
0.874331551,
0.887700535,
0.898395722,
0.885026738,
0.86631016,
0.868983957,
0.871657754,
0.877005348,
0.874331551,
0.893048128,
0.882352941,
0.882352941,
0.863636364,
0.895721925,
0.882352941,
0.871657754,
0.877005348,
0.885026738,
0.895721925,
0.893048128,
0.901069519,
0.885026738,
0.860962567,
0.919786096
]

f1_orig = [
0.768447837,
0.798882682,
0.820058997,
0.791139241,
0.808743169,
0.831908832,
0.838888889,
0.831908832,
0.834319527,
0.833819242,
0.812121212,
0.856353591,
0.828169014,
0.852272727,
0.843835616,
0.861271676,
0.846153846,
0.850152905,
0.824207493,
0.832844575,
0.831325301,
0.878186969,
0.872628726,
0.872727273,
0.892655367,
0.875362319,
0.86631016,
0.867208672,
0.857988166,
0.86627907,
0.871232877,
0.883040936,
0.874285714,
0.8625,
0.858725762,
0.891364903,
0.87283237,
0.862857143,
0.870056497,
0.869300912,
0.889518414,
0.88700565,
0.888217523,
0.876080692,
0.860962567,
0.913294798,
]
auc_orig = [
0.850768126,
0.87776023,
0.886891601,
0.89994257,
0.890078966,
0.904824121,
0.915190237,
0.911026561,
0.912175162,
0.927623833,
0.922254128,
0.938435032,
0.928413496,
0.937315147,
0.932534099,
0.94068916,
0.94384781,
0.952763819,
0.936625987,
0.932749462,
0.945915291,
0.949102656,
0.951658291,
0.963244795,
0.959698492,
0.958750897,
0.956066045,
0.952361809,
0.956468055,
0.958176597,
0.949806174,
0.963948313,
0.953912419,
0.968729361,
0.951284996,
0.967106963,
0.960502513,
0.954127782,
0.954271357,
0.965168701,
0.964149318,
0.971198851,
0.96149318,
0.96258435,
0.946733668,
0.979052405
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
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base + U_orig") #标题
# plt.savefig('../savedfig/CXC_ft_IN1K_mae_pretrain_vit_base_U_orig.png')
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
# plt.title("CXC_ft_IN1K_mae_pretrain_vit_base + U_sani") #标题
# # plt.show()
# plt.savefig('../savedfig/CXC_ft_IN1K_mae_pretrain_vit_base_U_sani.png')

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
# plt.title('CXCbase',fontsize=14)
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
# 设置主刻度标签的位置,标签文本的格式
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
 
# 显示次刻度标签的位置
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

plt.savefig('./linechart_CXCbase.png')