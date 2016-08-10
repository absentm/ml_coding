# coding=utf-8
# adaboost.py
# adaboost算法实现

from math import log
from numpy import matrix, ones, shape, mat, inf, zeros, array
from numpy.core.umath import multiply, exp, sign
import matplotlib.pyplot as plt


def loadSimpData():
    '''
    创建简单数据集
    :return: dataMat，classLabels
    '''
    dataMat = matrix([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return dataMat, classLabels


def loadDataSet(fileName):
    '''
    自适应数据加载函数，自动检测出数据集的特征数目
    :param fileName: 数据文件名
    :return: dataMat，数据集；labelMat，类别标签
    '''
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')

        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))

        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较进行分类
    :param dataMatrix: 原始数据集
    :param dimen: 尺寸
    :param threshVal: 阈值
    :param threshIneq: 阈值比较
    :return:
    '''
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    基于加权输入值进行决策的分类器：
    遍历stumpClassify()上的所有的可能输入值，并找到数据集上最佳的单层决策树
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param D: 权重向量
    :return: bestStump：词典，保存最佳单层决策树；
             minError：最小错误率
             bestClasEst：类别估计值
    '''
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # 置最小误差为 +oo

    for i in range(n):  # 遍历数据集的所以特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 通过计算最大值和最小值了解步长

        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # calc total error multiplied by D
                print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" \
                      % (i, threshVal, inequal, weightedError)

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    '''
    基于单层决策树的AdaBoost训练过程
    :param dataArr: 数据集
    :param classLabels: 类标签
    :param numIt: 迭代次数， 用户自定义指定
    :return: weakClassArr， 弱分类器集合；aggClassEst，每个数据点的类别估计累计值
    '''
    # 初始化
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 初始化概率分布向量，其元素之和为 1
    aggClassEst = mat(zeros((m, 1)))

    for i in range(numIt):
        # 构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:", D.T

        # alpha每个分类器配备的权重值， 计算公式：alpha = (1/2) * ln[(1-e) / e]
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 存储最佳决策树
        print "classEst: ", classEst.T

        # 更新权重向量D
        # 若正确分类，D[t + 1] = [D[t]*exp(-a) / sum(D)]
        # 若错误分类，D[t + 1] = [D[t]*exp(+a) / sum(D)]
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()

        aggClassEst += alpha * classEst  # 更新累计类别估计值
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m  # 计算错误率
        print "total error: ", errorRate

        if errorRate == 0.0:
            break  # 为0， 退出循环

    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    '''
    利用训练得到的多个弱分类器进行分类
    :param datToClass: 一个或多个待分类样例
    :param classifierArr: 多个弱分类器组成的数组
    :return:
    '''
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))

    for j in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[int(j)]['dim'], \
                                 classifierArr[int(j)]['thresh'], \
                                 classifierArr[int(j)]['ineq'])
        aggClassEst += classifierArr[int(j)]['alpha'] * classEst
        print aggClassEst

    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    '''
    使用ROC曲线度量分类器
    :param predStrengths: 数组，分类器的预测强度
    :param classLabels: 类标签
    :return:
    '''
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()  # 获取排好序的索引
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]

        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is: ", ySum * xStep
