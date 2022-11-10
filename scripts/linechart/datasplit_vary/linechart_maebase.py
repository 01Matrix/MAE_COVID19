# encoding=utf-8
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

acc_sani = [
0.606896552,
0.517241379,
0.682758621,
0.724137931,
0.682758621,
0.689655172,
0.737931034,
0.710344828,
0.75862069,
0.744827586,
0.806896552,
0.751724138,
0.772413793,
0.779310345,
0.75862069,
0.8,
0.75862069,
0.8,
0.8,
0.8,
0.772413793,
0.744827586,
0.8,
0.620689655,
0.765517241,
0.737931034,
0.806896552,
0.786206897,
0.820689655,
0.813793103,
0.786206897,
0.696551724,
0.696551724,
0.786206897,
0.820689655,
0.813793103,
0.786206897,
0.751724138,
0.717241379,
0.8,
0.813793103,
0.793103448,
0.8,
0.813793103,
0.875862069,
0.806896552
]
f1_sani = [
0.658682635,
0.646464646,
0.661764706,
0.696969697,
0.671428571,
0.705882353,
0.672413793,
0.588235294,
0.705882353,
0.741258741,
0.770491803,
0.735294118,
0.691588785,
0.780821918,
0.774193548,
0.791366906,
0.744525547,
0.778625954,
0.791366906,
0.791366906,
0.775510204,
0.758169935,
0.794326241,
0.69273743,
0.696428571,
0.728571429,
0.784615385,
0.786206897,
0.819444444,
0.773109244,
0.763358779,
0.728395062,
0.717948718,
0.77037037,
0.8,
0.79389313,
0.71559633,
0.647058824,
0.738853503,
0.8,
0.787401575,
0.769230769,
0.794326241,
0.805755396,
0.85483871,
0.810810811
]
auc_sani = [
0.721614883,
0.654583813,
0.744725738,
0.81482547,
0.798043728,
0.809647104,
0.815976218,
0.860375911,
0.836401995,
0.868910625,
0.901898734,
0.855677023,
0.865841964,
0.882719601,
0.883486766,
0.876582278,
0.852896049,
0.869294208,
0.860951285,
0.860471807,
0.87965094,
0.869102417,
0.870540852,
0.846758727,
0.888569237,
0.826908324,
0.883294975,
0.866225547,
0.880418105,
0.898158803,
0.887130802,
0.879171461,
0.837648638,
0.864691216,
0.891637898,
0.905926352,
0.882719601,
0.861143076,
0.8716916,
0.902282317,
0.890678941,
0.878883774,
0.888185654,
0.889624089,
0.945531262,
0.917337936
]

# # ***************************
acc_orig = [
0.735294118,
0.745989305,
0.794117647,
0.834224599,
0.815508021,
0.778074866,
0.815508021,
0.812834225,
0.786096257,
0.778074866,
0.826203209,
0.778074866,
0.810160428,
0.804812834,
0.831550802,
0.831550802,
0.802139037,
0.788770053,
0.812834225,
0.844919786,
0.812834225,
0.839572193,
0.863636364,
0.836898396,
0.85828877,
0.86631016,
0.828877005,
0.85828877,
0.810160428,
0.882352941,
0.86631016,
0.877005348,
0.681818182,
0.852941176,
0.874331551,
0.887700535,
0.847593583,
0.874331551,
0.890374332,
0.879679144,
0.882352941,
0.703208556,
0.919786096,
0.860962567,
0.901069519,
0.922459893
]

f1_orig = [
0.671096346,
0.666666667,
0.752411576,
0.817647059,
0.78369906,
0.79093199,
0.807799443,
0.808743169,
0.781420765,
0.79197995,
0.794952681,
0.736507937,
0.789317507,
0.802168022,
0.825484765,
0.817391304,
0.756578947,
0.805896806,
0.810810811,
0.815286624,
0.778481013,
0.823529412,
0.862533693,
0.812307692,
0.847262248,
0.852941176,
0.797468354,
0.85399449,
0.811671088,
0.879120879,
0.844720497,
0.861445783,
0.74186551,
0.845070423,
0.86908078,
0.879310345,
0.815533981,
0.872628726,
0.870662461,
0.873239437,
0.871345029,
0.736342043,
0.913793103,
0.834394904,
0.890207715,
0.916905444
]

auc_orig = [
0.819684135,
0.839152907,
0.884953338,
0.907910983,
0.905728643,
0.875434314,
0.881349605,
0.905513281,
0.886762383,
0.904393396,
0.909662599,
0.88051687,
0.90086145,
0.905642498,
0.927623833,
0.921708543,
0.923029433,
0.905599426,
0.89816224,
0.941866475,
0.925958363,
0.928011486,
0.941593683,
0.928011486,
0.939856425,
0.950007179,
0.925829146,
0.939971285,
0.919095477,
0.948557071,
0.945412778,
0.948140704,
0.912132089,
0.947121321,
0.944666188,
0.952376167,
0.946015793,
0.957257717,
0.974687724,
0.946719311,
0.952290022,
0.819583632,
0.965312276,
0.944479541,
0.959109835,
0.973338119
]

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
# plt.title('maebase',fontsize=14)
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

plt.savefig('./linechart_maebase.png')