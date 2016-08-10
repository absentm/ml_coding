# coding=utf-8
# main_test.py
# 主测试函数
from numpy import array

from Logistic_Regression.main import logRegres


def main_test():
    dataArr, labelMat = logRegres.loadDataSet()
    print "dataArr >>> ", dataArr
    print "labelMat >>> ", labelMat

    # 画图最佳拟合直线， 测试梯度上升算法
    weights0 = logRegres.gradAscent(dataArr, labelMat)
    print ">>> ", weights0
    logRegres.plotBestFit(weights0.getA())  # figure_2.png

    # 画图最佳拟合直线， 测试随机梯度上升算法
    weights1 = logRegres.stocGradAscent0(array(dataArr), labelMat)
    print ">>> ", weights1
    logRegres.plotBestFit(weights1)  # figure_3.png

    # 画图最佳拟合直线， 测试改进的随机梯度上升算法
    weights2 = logRegres.stocGradAscent0(array(dataArr), labelMat)
    print ">>> ", weights2
    logRegres.plotBestFit(weights2)  # figure_5.png

    # 测试马疝病数据集效果
    logRegres.multiTest()


if __name__ == '__main__':
    main_test()
