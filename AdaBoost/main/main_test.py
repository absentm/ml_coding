# coding=utf-8
# main_test.py
# 主测试函数
import os

from numpy import ones, mat

from AdaBoost.main import adaboost


def main_test():
    '''
    主测试函数
    :return:
    '''
    print "AdaBoost"

    # 加载数据
    dataMat, classLabels = adaboost.loadSimpData()
    print "dataMat >>> ", dataMat
    print "classLabels >>> ", classLabels

    # 测试buildStump()
    D = mat(ones((5, 1)) / 5)
    print "D >>> ", D

    bestStump, minError, bestClasEst = adaboost.buildStump(dataMat, classLabels, D)
    print "bestStump >>> ", bestStump
    print "minError >>> ", minError
    print "bestClasEst >>> ", bestClasEst

    # 测试adaBoostTrainDS()函数
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(dataMat, classLabels, 30)
    print type(classifierArray)
    print "classifierArray >>> ", classifierArray
    print "aggClassEst >>> ", aggClassEst

    # 简单测试算法
    classResults = adaboost.adaClassify([0, 0], classifierArray)
    print "classResult >>> ", classResults

    # 马疝气数据集上测试
    dataArr, labelArr = adaboost.loadDataSet(os.path.dirname(os.getcwd())
                                             + "\\datas\\horseColicTraining2.txt")
    classifierArr, aggEst = adaboost.adaBoostTrainDS(dataArr, labelArr, 10)
    print "classifierArr >>> ", classifierArr
    print "aggEst >>> ", aggEst

    testArr, testLabelArr = adaboost.loadDataSet(os.path.dirname(os.getcwd())
                                                 + "\\datas\\horseColicTest2.txt")
    print "testArr >>> ", testArr
    print "testLabelArr >>> ", testLabelArr

    prediction10 = adaboost.adaClassify(testArr, classifierArr)
    print "prediction10 >>> ", prediction10

    # 计算错误率
    errArr = mat(ones((len(dataArr), 1)))
    errTotal = errArr[prediction10 != mat(testLabelArr).T].sum()
    errorRates = errTotal / len(dataArr)
    print "errTotal >>> ", errTotal
    print "errorRates >>> ", errorRates

    # ROC曲线度量分类器
    adaboost.plotROC(aggEst.T, labelArr)


if __name__ == '__main__':
    main_test()
