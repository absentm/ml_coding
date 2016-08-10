# coding=utf-8
# main_test.py
# 主测试函数
import os

from numpy import eye, mat, array

from Tree_Regression.main import regTrees


def cartTest():
    '''
    分类回归树测试
    :return:
    '''
    testMat = mat(eye(4))
    print "testMat >>> ", testMat

    mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
    print "mat0 >>> ", mat0
    print "mat1 >>> ", mat1

    # myData = regTrees.loadDataSet(os.path.dirname(os.getcwd()) +
    #                               "\\datas\\ex00.txt")
    # print "myData >>> ", myData
    # myMat = mat(myData)
    # print "myMat >>> ", myMat
    # retTree = regTrees.createTree(myMat)
    # print "retTree >>> ", retTree


def main_test():
    '''
    # 主测试函数
    :return:
    '''
    print "Tree_Regression"
    cartTest()


if __name__ == '__main__':
    main_test()
