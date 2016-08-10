# coding=utf-8
# kMeans.py
# k-均值聚类算法

from numpy import *
import urllib
import json
from time import sleep
import matplotlib.pyplot as plt
import os


def loadDataSet(fileName):
    '''
    加载数据集
    :param fileName: 文件名
    :return:
    '''
    dataMat = []
    fr = open(fileName)

    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # 浮点化元素
        dataMat.append(fltLine)

    return dataMat


def distEclud(vecA, vecB):
    '''
    计算两个向量的欧式距离
    :param vecA: 向量A
    :param vecB: 向量B
    :return:
    '''
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    '''
    给定数据集，构建一个包含k个随机质心的集合
    :param dataSet: 数据集
    :param k: 参数，用户自定义
    :return: 簇质心集合
    '''
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # 构建簇质心

    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))

    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    K-均值聚类算法的实现
    :param dataSet: 数据集
    :param k: 簇个数， 用户自定义
    :param distMeas: 欧氏距离，可选
    :param createCent: 初始质心集合，可选
    :return: centroids, 质心集合； clusterAssment，簇分配结果矩阵
    '''
    m = shape(dataSet)[0]

    # clusterAssment簇分配结果矩阵（两列），一列记录簇索引值，另一列存储误差
    clusterAssment = mat(zeros((m, 2)))

    centroids = createCent(dataSet, k)  # 质心集合
    clusterChanged = True  # 标志变量，True则继续迭代

    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1

            # 寻找最近的质心
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])

                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        print centroids
        for cent in range(k):  # 更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)

    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    '''
    二分聚类算法的实现
    :param dataSet: 数据集
    :param k: 参数，用户自定义
    :param distMeas: 计算欧氏距离，可选
    :return: mat(centList), 质心列表； clusterAssment，簇分配结果
    '''
    # 创建初始簇
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]

    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2

    while (len(centList) < k):
        lowestSSE = inf

        for i in range(len(centList)):
            # 尝试划分每一簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])

            print "sseSplit, and notSplit: ", sseSplit, sseNotSplit

            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # 更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)

        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    return mat(centList), clusterAssment

# Yahoo 地图数据集应用
def geoGrab(stAddress, city):
    '''
    从 Yahoo API解析一些数据
    :param stAddress: 街道地址
    :param city: 城市的字符串
    :return:
    '''
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'  # 返回Json格式数据
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)

    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)

    return json.loads(c.read())


def massPlaceFind(fileName):
    '''
    从文件中获取数据，读取经纬度
    :param fileName: 文件名
    :return:
    '''
    fw = open(os.path.dirname(os.getcwd()) + '\\datas\\places.txt', 'w')

    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])

        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print "error fetching"
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):
    '''
    球面距离计算
    :param vecA: 向量A
    :param vecB: 向量B
    :return: 返回地球表面两点之间的距离
    '''
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)

    return arccos(a + b) * 6371.0


def clusterClubs(numClust=5):
    '''
    在地图上画出簇质心
    :param numClust: 簇数目，可选
    :return:
    '''
    datList = []
    for line in open(os.path.dirname(os.getcwd()) + '\\datas\\places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])

    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)

    # imread()基于图像创建矩阵，imshow()显示图像
    imgP = plt.imread(os.path.dirname(os.getcwd()) + '\\datas\\Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)

    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0],
                    ptsInCurrCluster[:, 1].flatten().A[0],
                    marker=markerStyle, s=90)

    ax1.scatter(myCentroids[:, 0].flatten().A[0],
                myCentroids[:, 1].flatten().A[0],
                marker='+', s=300)
    plt.show()
