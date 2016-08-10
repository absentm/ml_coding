# coding=utf-8
# plotter.py
# 数据可视化实现


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy import column_stack

# 设置中文显示
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)


def plotter(dataSet, centroids, clusterAssment):
    '''
    根据前向逐步回归系数与迭代次数对比图: figure_1.png
    :param redgeWeights: 岭回归系数集合
    :return:
    '''
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    xcord3 = []
    ycord3 = []
    xcord4 = []
    ycord4 = []

    classLabels = clusterAssment[:, 0]
    x = column_stack((classLabels, dataSet)).A  # 矩阵合并

    for i in range(len(x[:, 0])):
        if x[i, 0] == 0.0:
            xcord1.append(x[i, 1])
            ycord1.append(x[i, 2])
        if x[i, 0] == 1.0:
            xcord2.append(x[i, 1])
            ycord2.append(x[i, 2])
        if x[i, 0] == 2.0:
            xcord3.append(x[i, 1])
            ycord3.append(x[i, 2])
        if x[i, 0] == 3.0:
            xcord4.append(x[i, 1])
            ycord4.append(x[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    type1 = ax.scatter(xcord1, ycord1, s=50, c='red')
    type2 = ax.scatter(xcord2, ycord2, s=50, c='green')
    type3 = ax.scatter(xcord3, ycord3, s=50, c='blue')
    type4 = ax.scatter(xcord4, ycord4, s=50, c='black')

    ax.legend([type1, type2, type3, type4], ["1", "2", "3", "4"], loc=2)
    ax.plot(centroids[:, 0], centroids[:, 1], "k+")

    plt.title(u"K-均值聚类结果示意图", fontproperties=font)
    plt.xlabel(u"X 轴", fontproperties=font)
    plt.ylabel(u"Y 轴", fontproperties=font)
    plt.show()
