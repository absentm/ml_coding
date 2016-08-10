# coding=utf-8
# main_test.py
# 主测试函数

import os
from numpy import mat, corrcoef
from Regression.extras_plotter import plotter
from Regression.main import regression


def ex0Test():
    '''
    ex0.txt数据集上线性回归，局部加权线性回归测试
    :return:
    '''
    # 加载数据集
    xArr, yArr = regression.loadDataSet(os.path.dirname(os.getcwd()) +
                                        "\\datas\\ex0.txt")
    print "xArr >>> ", xArr
    print "yArr >>> ", yArr

    # standRegres()计算回归系数
    ws = regression.standRegres(xArr, yArr)
    print "ws >>> ", ws
    plotter.plotter1(xArr, yArr)  # 绘图，figure_1.png

    # 计算预测值和真实值的相关性
    xMat = mat(xArr)
    yMat = mat(yArr)  # 真实值
    yHat = xMat * ws  # 预测值
    corrcoefs = corrcoef(yHat.T, yMat)
    print "corrcoefs >>> ", corrcoefs

    # 局部加权线性回归单点测试
    print "yArr[0] >>> ", yArr[0]
    print "高斯核参数k = 1.0时", regression.lwlr(xArr[0], xArr, yArr, 1.0)
    print "高斯核参数k = 0.001时", regression.lwlr(xArr[0], xArr, yArr, 0.001)

    # 局部加权线性回归可视化
    plotter.plotter2(xArr, yArr)  # figure_2.png
    plotter.plotter3(xArr, yArr)  # figure_3.png


def abaloneTest():
    '''
    局部加权线性回归在鲍鱼年龄数据集上的测试
    :return:
    '''
    abX, abY = regression.loadDataSet(os.path.dirname(os.getcwd()) +
                                      "\\datas\\abalone.txt")

    yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

    print "数据集：0 - 99，k=0.1时，误差分析", regression.rssError(abY[0:99], yHat01.T)
    print "数据集：0 - 99，k=1时，误差分析", regression.rssError(abY[0:99], yHat1.T)
    print "数据集：0 - 99，k=10时，误差分析", regression.rssError(abY[0:99], yHat10.T)

    yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)

    print "数据集：100 - 199，k=0.1时，误差分析", regression.rssError(abY[0:99], yHat01.T)
    print "数据集：100 - 199，k=1时，误差分析", regression.rssError(abY[0:99], yHat1.T)
    print "数据集：100 - 199，k=10时，误差分析", regression.rssError(abY[0:99], yHat10.T)

    print "测试结果表明：使用较小的核，能得到较低的误差，但在不同的训练集上，未必是最好结果！"

    ws = regression.standRegres(abX[0:99], abY[0:99])
    yHat = mat(abX[100:199]) * ws
    print "简单线性回归，k=10时，误差分析", regression.rssError(abY[100:199], yHat.T.A)


def ridgeTest():
    '''
    岭回归测试及可视化处理（鲍鱼年龄数据集）
    :return:
    '''
    abX, abY = regression.loadDataSet(os.path.dirname(os.getcwd()) +
                                      "\\datas\\abalone.txt")
    ridgeWeights = regression.ridgeTest(abX, abY)
    plotter.plotter4(ridgeWeights)  # figure_4.png


def stageWiseTest():
    '''
    前向逐步线性回归（鲍鱼年龄数据集）
    :return:
    '''
    xArr, yArr = regression.loadDataSet(os.path.dirname(os.getcwd()) +
                                        "\\datas\\abalone.txt")
    wsMat, ws = regression.stageWise(xArr, yArr, 0.001, 2000)
    print "wsMat >>> ", wsMat
    print "ws >>> ", ws

    plotter.plotter5(wsMat)  # figure_5.png


def legoTest():
    '''
    乐高玩具套装测试
    :return:
    '''
    lgX = []
    lgY = []
    regression.setDataCollect(lgX, lgY)

    print "lgX >>> ", lgX
    print "lgY >>> ", lgY

    regression.crossValidation(lgX, lgY, 10)
    wMat = regression.ridgeTest(lgX, lgY)
    print "wMat >>> ", wMat


def main_test():
    '''
    主测试函数
    :return:
    '''
    print "Regression"
    ex0Test()  # ex0.txt数据集测试
    abaloneTest()  # 鲍鱼年龄数据集上测试
    ridgeTest()  # 在鲍鱼年龄数据集上测试岭回归
    stageWiseTest()  # 在鲍鱼年龄数据集上测试前向逐步线性回归
    # legoTest()  # 乐高玩具套装岭回归测试


if __name__ == '__main__':
    main_test()
