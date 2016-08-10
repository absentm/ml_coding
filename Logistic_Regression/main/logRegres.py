# coding=utf-8
# logRegres.py
# 逻辑回归(Logistic_Regression)算法的实现

import os
import random

from numpy import mat, shape, ones, array, arange
from numpy.core.umath import exp
import matplotlib.pyplot as plt


def loadDataSet():
    '''
    加载数据集, 并作简单转换
    :return:
    '''
    dataMat = []  # 数据
    lableMat = []  # 数据标签

    #  打开当前目录的上级目录\\datas\\testSet.txt
    fr = open(os.path.dirname(os.getcwd()) + "\\datas\\testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()

        # 第一个元素X0置为1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        lableMat.append(int(lineArr[2]))

    return dataMat, lableMat


def sigmoid(inX):
    '''
    Sigmoid函数
    :param inX:
    :return: 参数
    '''
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升优化算法的实现， 计算最大值
    :param dataMatIn: 数据集
    :param classLabels: 分类标签
    :return:
    '''
    # 将数据转化为Numpy矩阵数据类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()

    # 初始化变量
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weight = ones((n, 1))

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weight)
        error = (labelMat - h)
        weight = weight + alpha * dataMatrix.transpose() * error  # 更新回归系数

    return weight


def plotBestFit(weights):
    '''
    画出数据集和 Logistic回归zuiji9a你和直线
    :param weights: 回归系数
    :return:
    '''
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMat, classLabels):
    '''
    随机梯度上升算法
    :param dataMat: 数据集
    :param classLabels: 分类标签
    :return: 回归系数
    '''
    m, n = shape(dataMat)
    alpha = 0.01
    weights = ones(n)

    for i in range(m):
        h = sigmoid(sum(dataMat[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMat[i]

    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    改进随机梯度上升算法
    :param dataMatrix: 数据集
    :param classLabels: 类标签
    :param numIter:
    :return:
    '''
    m, n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # 每次迭代时改进, 避免参数的严格下降
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))

            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])

    return weights


def classifyVector(inX, weights):
    '''
    分类：sigmoid()函数值 > 0.5, 类标签置为 1；否则，置为 0
    :param inX: 特征向量
    :param weights: 回归系数
    :return: 类别标签
    '''
    prob = sigmoid(sum(inX * weights))

    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    '''
    对数据进行格式化处理
    :return:
    '''
    frTrain = open(os.path.dirname(os.getcwd()) + "\\datas\\horseColicTraining.txt")
    frTest = open(os.path.dirname(os.getcwd()) + "\\datas\\horseColicTest.txt")

    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1

    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    '''
    计算 10测试结果的平均值
    :return:
    '''
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))
