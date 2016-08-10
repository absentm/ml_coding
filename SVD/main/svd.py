# coding=utf-8
# svd.py
# SVD算法

from numpy import *
from numpy import linalg as la
import os


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def ecludSim(inA, inB):
    '''
    相似度计算法
    :param inA: 列向量A
    :param inB: 列向量B
    :return:
    '''
    return 1.0 / (1.0 + la.norm(inA - inB))  # norm()范数计算


def pearsSim(inA, inB):
    '''
    皮尔逊相关系数法
    :param inA: 列向量A
    :param inB: 列向量B
    :return:
    '''
    if len(inA) < 3: return 1.0  # 是否存在3个或更多的点
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    '''
    余弦相似度法
    :param inA: 列向量A
    :param inB: 列向量B
    :return:
    '''
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    '''
    在给定相似度计算方法的条件下，用户对物品的估计评分值
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 物品编号
    :return:
    '''
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
                                      dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], \
                                 dataMat[overLap, j])
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    '''
    推荐引擎，产生最高的N个推荐结果
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param N: 推荐结果个数
    :param simMeas: 相似度计算方法
    :param estMethod: 估计方法
    :return:
    '''
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # find unrated items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def svdEst(dataMat, user, simMeas, item):
    '''
    基于SVD的评分估计
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 物品条目
    :return:
    '''
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])  # 建立对角矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # 构建转化后的物品

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, \
                             xformedItems[j, :].T)

        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 基于SVD的图像压缩，降维处理
def printMat(inMat, thresh=0.8):
    '''
    打印矩阵
    :param inMat: 输入值
    :param thresh: 阈值
    :return:
    '''
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1,
            else:
                print 0,
        print ''


def imgCompress(numSV=3, thresh=0.8):
    '''
    模拟图像的压缩
    :param numSV:
    :param thresh:
    :return:
    '''
    myl = []
    for line in open(os.path.dirname(os.getcwd())
                             + '\\datas\\0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)

    myMat = mat(myl)
    print "****original matrix******"
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))

    for k in range(numSV):  # construct diagonal matrix from vector
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]

    print "****reconstructed matrix using %d singular values******" % numSV
    printMat(reconMat, thresh)
