# coding=utf-8
# regression.py
# 回归算法实现
import random

from numpy import mat, shape, eye, zeros, mean, var, inf, nonzero, array
from numpy.core.umath import exp, multiply
from numpy.linalg import linalg
from time import sleep
import json
import urllib2


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


def standRegres(xArr, yArr):
    '''
    使用普通最小二乘法(OSL，平方误差)，计算最佳拟合直线的回归系数：
        W = (X^T * X)^(-1) * X^T * y
    :param xArr: 给定数据 X
    :param yArr: 真实数据值 Y
    :return: 回归系数集合
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T  # 转置
    xTx = xMat.T * xMat  # 计算 X^T * X

    # 判断矩阵是否可逆，依据：方阵的行列式的值是否为0
    if linalg.det(xTx) == 0.0:
        print "该矩阵不可逆"
        return

    ws = xTx.I * (xMat.T * yMat)  # xTx.I 计算矩阵的逆矩阵
    # ws = linalg.solve(xTx, xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    局部加权线性回归，
    回归系数计算公式：w = (X^TWX)^(-1)X^TWy
    高斯核计算公式：w(i, i) = exp{[x^(i) - x] / (-2 * k^2)}
    :param testPoint: 坐标点
    :param xArr: 输入值
    :param yArr: 真实值
    :param k: 高斯核参数，用户自定义
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 初始化权重矩阵

    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))  # 高斯核

    xTx = xMat.T * (weights * xMat)

    if linalg.det(xTx) == 0.0:  # 判断矩阵是否可逆
        print "This matrix is singular, cannot do inverse"
        return

    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
    给定空间任意一点，局部加权计算对应的预测值
    :param testArr: 待测矩阵
    :param xArr: 输入值
    :param yArr: 真实值
    :param k: 参数
    :return: 预测值
    '''
    m = shape(testArr)[0]
    yHat = zeros(m)

    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)

    return yHat


def rssError(yArr, yHatArr):
    '''
    辅助函数，用于分析误差大小
    :param yArr: 真实值
    :param yHatArr: 预测值
    :return:
    '''
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    岭回归：处理特征数多于样本数的情况，引入lam*eye(m)使得矩阵非奇异
    ws = (X^TX + lam*eye(m)).I * X^T * y
    :param xMat: 输入值
    :param yMat: 真实值
    :param lam: 参数 lam
    :return: 回归系数 ws
    '''
    xTx = xMat.T * xMat
    demo = xTx + eye(shape(xMat)[1]) * lam

    if linalg.det(demo) == 0.0:  # 判断是否可逆
        print "This matrix is singular, cannot do inverse"
        return

    ws = demo.I * (xMat.T * yMat)

    return ws


def ridgeTest(xArr, yArr):
    '''
    计算30不同的参数lam 所对应的回归系数
    数据标准化处理：所有的特征都减去各自的均值并除以方差；
    :param xArr: 输入值
    :param yArr: 真实值
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T

    # mean()计算均值
    xMean = mean(xMat, 0)
    yMean = mean(yMat, 0)

    xVar = var(xMat, 0)  # var() 计算方差
    xMat = (xMat - xMean) / xVar
    yMat = yMat - yMean

    numTestPts = 30  # 迭代次数
    wMat = zeros((numTestPts, shape(xMat)[1]))  # 返回矩阵

    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))  # 计算回归系数，指数级
        wMat[i, :] = ws.T

    return wMat


def regularize(xMat):
    '''
    数据标准化，均值为0，方差为1
    :param xMat: 输入待标准化数据
    :return:
    '''
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar

    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    前向逐步线性回归
    :param xArr: 输入值
    :param yArr: 预测值
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T

    # 数据标准化处理，均值为0、方差为1
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)

    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):
        print ws.T
        lowestError = inf  # 设置当前最小误差为+oo

        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)

                # 如果误差小于当前最小误差，更新w
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest

        ws = wsMax.copy()
        returnMat[i, :] = ws.T

    return returnMat, ws


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    '''
    使用Google API获取网站购物信息
    :param retX: 产品的特征
    :param retY: 售价
    :param setNum: 网页编号
    :param yr: 年份
    :param numPce: 总价
    :param origPrc: 原价
    :return:
    '''
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())

    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']

            for item in listOfInv:
                sellingPrice = item['price']

                if sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print 'problem with item %d' % i


def setDataCollect(retX, retY):
    '''
    获取数据
    :param retX:
    :param retY:
    :return:
    '''
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    '''
    交叉验证测试岭回归
    :param xArr:
    :param yArr:
    :param numVal:
    :return:
    '''
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))

    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)

        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])

        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
            print errorMat[i, k]

    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]

    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX

    print "the best model from Ridge Regression is:\n", unReg
    print "with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)
