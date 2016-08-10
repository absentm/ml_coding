# coding=utf-8
# knn.py

'''
K-近邻算法

基本思想：
    存在一个样本数据集合（训练样本集），并且样本集中每个数据都存在标签，即
    我们已经知道每个数据与其对应类的关系。输入没有标签的新数据后，将新数据
    的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本中特征最相
    似数据（最近邻）的分类标签。通常，只选取样本集中前 K个最相似的数据, 也
    即（K <= 20, 整数），最后，选择 K个最相似数据中出现次数最多的分类，作
    为新数据的分类。
'''

from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir


def createDataSet():
    '''
    生成数据集和类标签
    :return: 数据集和类标签
    '''

    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 二维矩阵
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    使用欧式距离公式实现 k-近邻, d = [(x1 - x0)^2 + (y1 - y2)^2]^0.5
    :param inX: 待测输入向量
    :param dataSet: 训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻的数目
    :return: 待测数据的分类标签
    '''
    dataSetSize = dataSet.shape[0]  # 数据集的行数
    # print "dataSetSize =", dataSetSize
    # print "tile(inX, (dataSetSize, 1)) = ", tile(inX, (dataSetSize, 1))

    # tile复制inX生成与dataSet同形的矩阵, 计算距离矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 距离矩阵
    # print "diffMat = ", diffMat

    sqDiffMat = diffMat ** 2
    # print "sqDiffMat = ", sqDiffMat

    sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
    # print "sqDistances = ", sqDistances

    distances = sqDistances ** 0.5  # 开平方， 求得最终的距离向量
    # print "distances = ", distances

    sortedDistIndicies = distances.argsort()  # argsort()返回数组值从小到大的索引值
    # print "sortedDistIndicies = ", sortedDistIndicies

    classCount = {}  # 存放结果的字典

    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # print classCount
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def file2Matrix(filename):
    '''
    将文件内容转化为矩阵
    :param filename: 文件名
    :return: 二维矩阵，类标签
    '''
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)  # 获取文件内容的行数

    returnMat = zeros((numberOfLines, 3))  # 转化结果矩阵
    classLabelVector = []  # 类标签

    index = 0
    for line in arrayOfLines:
        line = line.strip()  # 去除串头和串尾的空白符
        listFromLine = line.split("\t")  # 以"\t"为分隔符，对行数据切片

        returnMat[index, :] = listFromLine[0:3]  # 按行转化为二维矩阵
        # print listFromLine[-1]
        classLabelVector.append(int(lastValueFormat(listFromLine[-1])))

        index += 1

    fr.close()
    return returnMat, classLabelVector


def lastValueFormat(str):
    '''
    字符串格式的偏好转化为数字
    :param str: 字符串
    :return: 数字 1, 2, 3
    '''

    if str == "largeDoses":
        return 3

    if str == "smallDoses":
        return 2

    if str == "didntLike":
        return 1


def make_image(dataMat, dataLabel):
    '''
    画图：生成散点图
    :param dataMat:
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(dataMat[:, 1], dataMat[:, 2])
    ax.scatter(dataMat[:, 1], dataMat[:, 2], 15 * array(dataLabel), 15 * array(dataLabel))
    plt.show()


def autoNorm(dataSet):
    '''
    归一化数值：newValue = (oldValue - min) / (max - min)
    :param dataSet: 矩阵
    :return: 规则化后矩阵
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]

    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    '''
    测试分类器性能
    :return:
    '''
    hoRate = 0.10
    datingMat, datingLabels = file2Matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingMat)  # 归一化

    m = normMat.shape[0]
    numTestVecs = int(m * hoRate)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "classifier result: %d, and Real: %d" % (classifierResult, datingLabels[i])

        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print "The error rate: %f" % (errorCount / float(numTestVecs))


def classifyPerson():
    '''
    约会网站预测函数
    :return:
    '''
    resultList = ["didntLike", "smallDoses", "largeDoses"]

    gamePers = float(raw_input("Input games time: "))
    malesPers = float(raw_input("Input mails: "))
    iceCreamPers = float(raw_input("inPut iceCream: "))

    datingMat, datingLabels = file2Matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingMat)  # 归一化

    inArr = array([gamePers, malesPers, iceCreamPers])
    result = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)

    print "You will probably like this person: ", resultList[result - 1]


def img2Vector(filename):
    '''
    工具函数：将图像转化为向量（1 * 1024）
    :param filename: 文件名称
    :return: 1 * 1024的向量
    '''
    returnVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()  # 每次只读一行文本
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])
    return returnVector


def handWritingClassTest():
    '''
    手写数字识别系统的测试
    :return:
    '''
    hwLabels = []

    trainingFileList = listdir("digits/trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 获取文件名
        fileStr = fileNameStr.split(".")[0]  # 去除文件名中的后缀
        classNumStr = int(fileStr.split("_")[0])  # 获取当前文件的分类标签
        hwLabels.append(classNumStr)  # 存储分类标签

        trainingMat[i, :] = img2Vector("digits/trainingDigits/%s" % fileNameStr)

    errorCount = 0.0
    testFileList = listdir("digits/testDigits")
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]  # 获取文件名
        fileStr = fileNameStr.split(".")[0]  # 去除文件名中的后缀
        classNumStr = int(fileStr.split("_")[0])  # 获取当前文件的分类标签

        vectorUnderTest = img2Vector("digits/testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "The classify Result: %d, Real: %d" % (classifierResult, classNumStr)

        if classifierResult != classNumStr:
            errorCount += 1.0

    # 输出结果
    print "\nTotal number of errors: %d" % errorCount
    print "\nTotal error rate: %f" % (errorCount / float(mTest))


def main_test():
    '''
    主测试函数
    :return:
    '''
    group, labels = createDataSet()
    print "group = ", group
    print "labels = ", labels

    print "[0, 0]测试结果： ", classify0([0, 0], group, labels, 3)

    datingMat, datingLabels = file2Matrix("datingTestSet.txt")
    print datingMat
    print datingLabels

    make_image(datingMat, datingLabels)  # 画图

    normMat, ranges, minVals = autoNorm(datingMat)  # 归一化
    print normMat
    print ranges
    print minVals

    datingClassTest()  # 测试分类器性能
    classifyPerson()  # 分类器在约会网站上的应用

    testVector = img2Vector("digits/testDigits/0_0.txt")
    print type(testVector)
    print testVector.shape[0]
    print testVector.shape[1]

    handWritingClassTest()  # 手写数字识别系统测试


if __name__ == '__main__':
    main_test()
