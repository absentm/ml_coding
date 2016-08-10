# coding=utf-8
# svmMLiA.py
# SVM算法实现：SMO(Sequential Minimal Optimization)序列最小化算法
import os
import random
from dircache import listdir
from numpy import mat, shape, zeros, nonzero
from numpy.core.umath import multiply, exp, sign


def loadDataSet(filename):
    '''
    加载并处理数据文件
    :param filename: 文件名称
    :return: 数据矩阵和类标签
    '''
    dataMat = []
    labelMat = []

    fr = open(filename)
    for line in fr.readlines():  # 逐行读文件
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat, labelMat


def selectJrand(i, m):
    '''
    随机下标选择
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数目
    :return:
    '''
    j = i
    while (j == i):
        j = int(random.uniform(0, m))

    return j


def clipAlpha(aj, h, l):
    '''
    辅助函数调整 alpha的值
    :param aj:
    :param h:
    :param l:
    :return:
    '''
    if aj > h:
        aj = h

    if l > aj:
        aj = l

    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    简化版 SMO算法实现
    :param dataMatIn: 数据集
    :param classLabels: 类标签
    :param C: 常数 C
    :param toler: 容错率
    :param maxIter: 取消前最大的循环次数
    :return:
    '''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()

    # 初始化
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0

    while (iter < maxIter):
        # 如果Alpha可以更改进入优化过程
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T *
                        (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])

            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 随机选择第二个alpha
                fXj = float(multiply(alphas, labelMat).T *
                            (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 保证alpha在0和C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print "L==H"
                    continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print "eta>=0"
                    continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) \
                              * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * \
                     dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * \
                     dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)

        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b, alphas


# 完整版Platt SMO的支持函数
class optStruct:
    '''
    建立数据结构保存所有重要的值
    '''

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        '''
        只包含一个构造函数，用于值的存储
        :param dataMatIn: 数据集
        :param classLabels: 类标签
        :param C: 常数C
        :param toler:
        :param kTup:
        :return:
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # 误差缓存
        self.K = mat(zeros((self.m, self.m)))

        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    '''
    对于给定的 alpha, 计算 E值
    :param oS:
    :param k:
    :return:
    '''
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    '''
    选择第二个 alpha值，内层循环的 alpha值
    :param i:
    :param oS:
    :param Ei:
    :return:
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 构建一个非零表

    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)

            # 选择具有最大步长的 j
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek

        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)

    return j, Ej


def updateEk(oS, k):
    '''
    辅助函数，计算误差值并缓存
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    '''
    完整版优化函数
    :param i:
    :param oS:
    :return:
    '''
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
                (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # 第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if L == H:
            print "L==H"
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel

        if eta >= 0:
            print "eta>=0"
            return 0

        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # 跟新缓存误差

        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough"
            return 0

        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)  # 跟新缓存误差
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''
    完整的SMO算法实现
    :param dataMatIn: 数据集
    :param classLabels: 类标签
    :param C: 常数 C
    :param toler:
    :param maxIter:
    :param kTup: 包含核函数信息的元组
    :return:
    '''
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        if entireSet:  # 便利所有值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" \
                      % (iter, i, alphaPairsChanged)
            iter += 1
        else:  # 遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" \
                      % (iter, i, alphaPairsChanged)
            iter += 1

        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True

        print "iteration number: %d" % iter

    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    '''
    计算 w值
    :param alphas: 参数
    :param dataArr: 数据集
    :param classLabels: 类标签
    :return:
    '''
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))

    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)

    return w


def kernelTrans(X, A, kTup):
    '''
    高斯径向基核函数，将数据从一个低维特征空间转换到高维特征空间
    K(x, y) = exp{[-(||x - y||)^2] / (2 * &^2}
    :param X: 数值型变量1
    :param A: 数值型变量2
    :param kTup: 元组
    :return:
    '''
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  #
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  #
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')

    return K


# 测试figure_3.png圆形样式数据集
def testRbf(k1=1.3):
    '''
    利用核函数进行分类的径向基测试函数, 数据集表现为圆形分类，如figure_3.png
    :param k1: 输入参数，高斯径向基中的用户自定义变量
    :return:
    '''
    dataArr, labelArr = loadDataSet(os.path.dirname(os.getcwd()) +
                                    '\\datas\\testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]  # 获取唯一的支持向量
    labelSV = labelMat[svInd]

    print "there are %d Support Vectors" % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0

    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print "the training error rate is: %f" % (float(errorCount) / m)

    dataArr, labelArr = loadDataSet(os.path.dirname(os.getcwd()) +
                                    '\\datas\\testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)

    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print "the test error rate is: %f" % (float(errorCount) / m)


# 测试手写数字识别

def img2vector(filename):
    '''
    工具函数：将图像转化为向量（1 * 1024）
    :param filename: 文件名称
    :return: 1 * 1024的向量
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])

    return returnVect


def loadImages(dirName):
    '''
    处理手写体数据集
    :param dirName: 目录名
    :return:
    '''
    hwLabels = []
    trainingFileList = listdir(dirName)  # 加载训练集
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)

        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))

    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    '''
    测试基于SVM的手写数字识别系统
    :param kTup: 输入参数，元组类型
    :return:
    '''
    dataArr, labelArr = loadImages(os.path.dirname(os.getcwd()) +
                                   '\\datas\\digits\\trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]

    print "there are %d Support Vectors" % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0

    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print "the training error rate is: %f" % (float(errorCount) / m)

    dataArr, labelArr = loadImages(os.path.dirname(os.getcwd()) +
                                   '\\datas\\digits\\testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)

    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print "the test error rate is: %f" % (float(errorCount) / m)
