# coding=utf-8
# main_test.py
# 主测试函数
import os

from numpy import shape, mean, cov, mat, argsort
from numpy.linalg import linalg

from PCA.extras_plotter import plotter
from PCA.main import pca


def simple_test():
    '''
    简单数据集上的测试
    :return:
    '''
    datMat = pca.loadDataSet(os.path.dirname(os.getcwd())
                             + "\\datas\\testSet.txt")
    lowDataMat, reconMat = pca.pca(datMat, 1)
    print "shape(lowDataMat) >>> ", shape(lowDataMat)

    plotter.plotter1(datMat, reconMat)  # figure_1.png


def decomTest():
    '''
    半导体制造数据集测试，及可视化
    :return:
    '''
    dataMat = pca.replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)  # 计算均值
    meanRemoved = dataMat - meanVals  # 去除均值
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算协方差矩阵的特征值和特征向量
    print "eigVals >>> ", eigVals

    eigValInd = argsort(eigVals)  #
    eigValInd = eigValInd[::-1]  #
    sortedEigVals = eigVals[eigValInd]
    total = sum(sortedEigVals)
    varPercentage = sortedEigVals / total * 100

    plotter.plotter2(varPercentage)  # figure_2.png


def main_test():
    print "PCA"
    simple_test()
    decomTest()


if __name__ == '__main__':
    main_test()
