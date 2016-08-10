# coding=utf-8
# main_test.py
# 主测试程序
import os

from numpy import mat

from K_Means.extras_plotter import plotter
from K_Means.main import kMeans


def kmeansTest():
    '''
    K-Means算法测试
    :return:
    '''

    # 加载数据集，测试数据集testSet.txt
    dataSet = kMeans.loadDataSet(os.path.dirname(os.getcwd()) +
                                 "\\datas\\testSet.txt")
    dataMat = mat(dataSet)
    print "dataMat >>> ", dataMat
    print "min(dataMat[:, 0]) >>> ", min(dataMat[:, 0])
    print "min(dataMat[:, 1]) >>> ", min(dataMat[:, 1])
    print "max(dataMat[:, 0]) >>> ", max(dataMat[:, 0])
    print "max(dataMat[:, 1]) >>> ", max(dataMat[:, 1])

    disVec = kMeans.distEclud(dataMat[0], dataMat[1])
    print "disVec >>> ", disVec

    centroids = kMeans.randCent(dataMat, 2)  # 生成随机质心集合
    print "centroids >>> ", centroids

    # 测试K-Means
    myCentroids, clusterAssment = kMeans.kMeans(dataMat, 4)
    print "myCentroids >>> ", myCentroids
    print "clusterAssment >>> ", clusterAssment

    plotter.plotter(dataMat, myCentroids, clusterAssment) # figure_1.png


def biKmeansTest():
    '''
    二分K-均值聚类算法实现
    :return:
    '''
    dataMat = mat(kMeans.loadDataSet(os.path.dirname(os.getcwd()) +
                                     "\\datas\\testSet2.txt"))

    centList, myNewAssments = kMeans.biKmeans(dataMat, 3)
    print "centList >>> ", centList
    print "myNewAssments >>> ", myNewAssments

    # geoResult = kMeans.geoGrab("1 VA Center", "Augusta, ME") 不能使用
    # print "geoResult >>> ", geoResult
    kMeans.clusterClubs(5)


def main_test():
    '''
    主程序
    :return:
    '''
    print "k_Means"
    kmeansTest()
    biKmeansTest()


if __name__ == '__main__':
    main_test()
