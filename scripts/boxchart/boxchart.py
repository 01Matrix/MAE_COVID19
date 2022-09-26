import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

if __name__ == '__main__':
    AA=[5,6,7,8,9,10,3]
    BB=[9,14,10,13,12,11,10]

    AA = {'A': AA}
    BB = {'B': BB}
    df1 = pd.DataFrame(AA)
    df2 = pd.DataFrame(BB)
    plt.grid(linestyle="--", alpha=0.3)
    plt.tick_params(labelsize=20)
    plt.xticks(rotation=10)   
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 25,
    #          }
    plt.boxplot(x=df1,        
                showmeans=True,  # 以点的形式显示均值
                positions=[1],
                #  boxprops={'color': 'black', 'facecolor': '#9999ff'},  # 设置箱体属性，填充色和边框色

                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},  # 设置异常值属性，点的形状、填充色和边框色

                meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色

                medianprops={'linestyle': '--', 'color': 'red'})  # 设置中位数线的属性，线的类型和颜色

    plt.boxplot(x=df2,
                patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充
                showmeans=True,  # 以点的形式显示均值
                positions=[2],
                boxprops={'color': 'black', 'facecolor': '#9999ff'},  # 设置箱体属性，填充色和边框色

                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},  # 设置异常值属性，点的形状、填充色和边框色

                meanprops={'marker': 'd', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色

                medianprops={'linestyle': '--', 'color': 'red'})  # 设置中位数线的属性，线的类型和颜色

    plt.xticks([1, 2],
               ['Alg1', 'Alg2'])
    plt.ylabel("Target", fontsize=30)
    # plt.ylabel("$obj$", fontsize=30)
    # plt.show()
    plt.savefig('./boxchart.png', bbox_inches='tight')
