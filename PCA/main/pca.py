# coding=utf-8
# pca.py
# PCA算法实现

from numpy import *
import os


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    '''
    PCA算法的实现
    :param dataMat: 数据集
    :param topNfeat: 应用的N个特征，可选
    :return: lowDDataMat, 降维后的数据集； reconMat，被重构后的原始数据集
    '''
    meanVals = mean(dataMat, axis=0)  # 计算均值
    meanRemoved = dataMat - meanVals  # 去除均值
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算协方差矩阵的特征值和特征向量

    eigValInd = argsort(eigVals)  # 排序，从小到大
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 去除多余的维度

    redEigVects = eigVects[:, eigValInd]  # 重组特征向量从大到小
    lowDDataMat = meanRemoved * redEigVects  # 将数据转化到新的空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals

    return lowDDataMat, reconMat


def replaceNanWithMean():
    '''
    将NaN替换成平均值
    :return:
    '''
    datMat = loadDataSet(os.path.dirname(os.getcwd())
                         + '\\datas\\secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 计算所有非NaN的平均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # 将所有NaN置为平均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat
