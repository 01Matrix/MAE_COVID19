# encoding=utf-8
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import make_interp_spline

acc_sani_cxclarge = [

]
f1_sani_cxclarge = [

]
auc_sani_cxclarge = [

]

acc_orig_cxclarge = [
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

acc_orig_data14large = [
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

acc_orig_maelarge = [
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

def smooth_xy(lx, ly):
    """数据平滑处理
    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]

x_values=[x/10 for x in range(4,50,1)]
xy_s1 = smooth_xy(x_values,acc_orig_maelarge)
xy_s2 = smooth_xy(x_values,acc_orig_data14large)
xy_s3 = smooth_xy(x_values,acc_orig_cxclarge)
plt.plot(xy_s1[0],xy_s1[1],label='MAE-L/16_IN1K')
plt.plot(xy_s2[0],xy_s2[1],label='MAE-L/16_Data14')
plt.plot(xy_s3[0],xy_s3[1],label='MAE-L/16_IN1K+CXC')
# plt.plot(x_values,acc_sani_cxclarge,label='U_sani')
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

plt.savefig('./linechart_compare_gap_large.png')