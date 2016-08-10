# coding=utf-8
# regTrees.py
# 树回归

from numpy import nonzero, shape, inf, mean, var, mat


def loadDataSet(fileName):
    '''
    加载数据
    :param fileName: 文件名
    :return:
    '''
    dataMat = []
    fr = open(fileName)

    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # 将每行数据映射为浮点数
        dataMat.append(fltLine)

    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''
    将数据集合切分并得到两个子集
    :param dataSet: 待切分集合
    :param feature: 数据特征
    :param value: 特征值
    :return: 两个子集
    '''
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]

    return mat0, mat1


def regLeaf(dataSet):
    '''
    生成叶节点，获取目标变量的均值
    :param dataSet: 数据集
    :return:
    '''
    return mean(dataSet[:, -1])


def regErr(dataSet):
    '''
    误差计算
    :param dataSet: 数据集, 获取目标变量的平方误差
    :return:
    '''
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    获取数据集切分的最佳二元切分位置
    :param dataSet: 数据集
    :param leafType: 建立叶节点的函数，可选
    :param errType: 误差计算函数，可选
    :param ops: 其他参数，可选，用户自定义
    :return:
    '''
    tolS = ops[0]
    tolN = ops[1]

    # 如果所有的目标变量相同，退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        return None, leafType(dataSet)

    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0

    # 遍历所有特征值
    for featIndex in range(n - 1):
        for splitVal in dataSet[:, featIndex]:
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)

            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue

            newS = errType(mat0) + errType(mat1)

            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    # 如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)

    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)

    # 如果切分出的数据集很小则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)

    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    递归创建分类回归树(CART)
    :param dataSet: 数据集
    :param leafType: 建立叶节点的函数，可选
    :param errType: 误差计算函数，可选
    :param ops: 其他参数，可选
    :return:
    '''
    # 获取最佳切分
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)

    if feat == None:
        return val

    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val

    lSet, rSet = binSplitDataSet(dataSet, feat, val)

    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree
