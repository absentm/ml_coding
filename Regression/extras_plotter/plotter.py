# coding=utf-8
# plotter.py
# 数据可视化实现


from numpy import mat
import matplotlib.pyplot as plt
from Regression.main import regression
from matplotlib.font_manager import FontProperties

# 设置中文显示
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)


def plotter1(xArr, yArr):
    '''
    数据集ex0上的线性回归最佳拟合可视化处理: figure_1.png
    :return:
    '''
    # # 加载数据集
    # xArr, yArr = regression.loadDataSet(os.path.dirname(os.getcwd()) +
    #                                     "\\datas\\ex0.txt")
    # ws = regression.standRegres(xArr, yArr)  # 回归系数

    xMat = mat(xArr)
    yMat = mat(yArr)  # 真实值

    # 创建图像（散点图），并绘制原始图像
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

    # 绘制拟合直线
    xCopy = xMat.copy()
    xCopy.sort(0)  # 防止数据点混乱，按升序排列
    yHat = xCopy * regression.standRegres(xArr, yArr)
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


def plotter2(xArr, yArr, k=0.01):
    '''
    局部加权线性回归可视化: figure_2.png
    :param xArr: 输入值
    :param yArr: 真实值
    :param k: 高斯核函数参数
    :return:
    '''
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)  # 数据点按升序排列
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()

    yHat = regression.lwlrTest(xArr, xArr, yArr, k)  # 所有点的估计值
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).flatten().A[0], s=2, c="red")

    plt.show()


def plotter3(xArr, yArr):
    '''
    局部加权线性回归可视化对比图: figure_3.png
    :param xArr: 输入值
    :param yArr: 真实值
    :return:
    '''
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)  # 数据点按升序排列
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()

    yHat1 = regression.lwlrTest(xArr, xArr, yArr, 1.0)  # 所有点的估计值, k=1.0
    ax = fig.add_subplot(311)
    ax.plot(xSort[:, 1], yHat1[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).flatten().A[0], s=2, c="red")
    plt.title("k = 1.0")

    yHat2 = regression.lwlrTest(xArr, xArr, yArr, 0.01)  # 所有点的估计值, k=0.01
    ax = fig.add_subplot(312)
    ax.plot(xSort[:, 1], yHat2[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).flatten().A[0], s=2, c="red")
    plt.title("k = 0.01")

    yHat3 = regression.lwlrTest(xArr, xArr, yArr, 0.003)  # 所有点的估计值, k=0.003
    ax = fig.add_subplot(313)
    ax.plot(xSort[:, 1], yHat3[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).flatten().A[0], s=2, c="red")
    plt.title("k = 0.003")

    plt.show()


def plotter4(ridgeWeights):
    '''
    根据岭回归系数，做对比图: figure_4.png
    :param redgeWeights: 岭回归系数集合
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.title(u"回归系数与log(lambda)的关系图", fontproperties=font)
    plt.xlabel("log(lambda)")
    plt.ylabel(u"回归系数", fontproperties=font)
    plt.show()


def plotter5(stageWiseWeights):
    '''
    根据前向逐步回归系数与迭代次数对比图: figure_5.png
    :param redgeWeights: 岭回归系数集合
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(stageWiseWeights)
    plt.title(u"前向逐步回归系数与迭代次数的关系图", fontproperties=font)
    plt.xlabel(u"迭代次数", fontproperties=font)
    plt.ylabel(u"回归系数", fontproperties=font)
    plt.show()
