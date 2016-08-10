# coding=utf-8
# trees.py
# 决策树算法

from math import log
import operator

from Decision_Tree import treePlotter


def createDataSet():
    '''
    生成数据集
    :return: 数据集， 类标签
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    '''
    计算香农熵， 计算公式为 H = -E{p[x(i)] * log2p[x(i)]]} (E为连加, 1 =< i <= n, n 为分类数目)
    :param dataSet: 数据集
    :return: 香农熵
    '''
    numEntries = len(dataSet)
    labelsCounts = {}

    for featVec in dataSet:
        print  "featVec >>> ", featVec
        currentLabel = featVec[-1]  # 获取当前行的最后一个元素，即标签
        print  "currentLabel >>> ", currentLabel

        # 计算各个标签出现的次数，存入labelsCounts中
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelsCounts:
        prob = float(labelsCounts[key]) / numEntries  # 计算每个分类标签的概率
        shannonEnt -= prob * log(prob, 2)

    print labelsCounts
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 待划分数据集的特征
    :param value: 特征的返回值
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        print "featVec >>> ", featVec
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # print "reducedFeatVec1 >>> ", reducedFeatVec
            reducedFeatVec.extend(featVec[axis + 1:])
            # print "reducedFeatVec2 >>> ", reducedFeatVec
            retDataSet.append(reducedFeatVec)

    print "reducedFeatVec >>> ", reducedFeatVec
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    :param dataSet: 数据集
    :return: 划分数据集最好方式下的特征
    '''
    numFeatures = len(dataSet[0]) - 1  # 最后一列，即类标签
    baseEntropy = calcShannonEnt(dataSet)  # 计算香农熵
    bestInfoGain = 0.0;
    bestFeature = -1

    for i in range(numFeatures):  # 迭代所有的特征
        featList = [example[i] for example in dataSet]  # 生成含有这个特征的所有实例的列表
        uniqueVals = set(featList)  # 去除重复的元素

        # 计算每种划分方式的信息熵
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy  # 计算信息增益

        # 计算最好的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature  # 返回一个整数


def majorityCnt(classList):
    '''
    计算出现次数最多的分类名称，确定叶子节点的分类
    :param classList: 类标签列表
    :return: 出现次数最多的分类名称
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1

    # 排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''
    递归创建树
    :param dataSet: 数据集
    :param labels: 标签列表
    :return: 决策树的结构
    '''
    classList = [example[-1] for example in dataSet]

    # 当类别完全相同时，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 遍历完所有的特征时， 返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)  # 计算最好的数据集划分方式
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}

    # 得到列表包含的所有属性值
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


def classify(inputTree, featLabels, testVec):
    '''
    使用决策树执行分类
    :param inputTree: 树结构字典
    :param featLabels: 类标签
    :param testVec: 待测向量
    :return: 分类结果，标签
    '''
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat

    return classLabel


def storeTree(inputTree, filename):
    '''
    使用 pickle模块存储决策树，序列化对象
    :param inputTree: 树结构
    :param filename: 文件名
    :return:
    '''
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    '''
    从文本中获取树的结构内容
    :param filename: 文件名称
    :return:
    '''
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def main_test():
    '''
    主测试函数
    :return:
    '''
    myData, labels = createDataSet()
    print "myData >>> ", myData
    print "labels >>> ", labels

    shannonEnt = calcShannonEnt(myData)
    print "shannonEnt >>> ", shannonEnt
    print splitDataSet(myData, 0, 0)

    print "Best Choose: ", chooseBestFeatureToSplit(myData)

    # myTree = createTree(myData, labels)
    # print "myTree >>> ", myTree

    myTree = treePlotter.retrieveTree(0)
    print "myTree >>> ", myTree
    print "测试分类结果1：", classify(myTree, labels, [1, 0])
    print "测试分类结果2：", classify(myTree, labels, [1, 1])

    storeTree(myTree, "classifierStorage.txt")
    print "测试序列化内容：", grabTree("classifierStorage.txt")


if __name__ == '__main__':
    main_test()
