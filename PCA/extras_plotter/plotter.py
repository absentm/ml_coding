# coding=utf-8
# plotter.py
# 可视化操作

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文显示
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)


def plotter1(dataMat, reconMat):
    '''
    绘制降维对比图
    :param dataMat: 原始数据集
    :param reconMat: 降维后数据集
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0],
               dataMat[:, 1].flatten().A[0],
               marker="^", s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0],
               reconMat[:, 1].flatten().A[0],
               marker="o", s=50, c="red")
    plt.title(u"原始数据集（三角点表示）及第一主成分（圆形点表示）", fontproperties=font)
    plt.xlabel(u"X 轴", fontproperties=font)
    plt.ylabel(u"Y 轴", fontproperties=font)
    plt.show()


def plotter2(varPercentage):
    '''
    半导体制造数据集测试，及可视化 figure_2.png
    :param dataMat:
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 21), varPercentage[:20], marker='^')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Percentage of Variance')
    plt.show()  # figure_2.png
