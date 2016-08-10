# coding=utf-8
# main_test.py
# 主测试函数

import os

from SVM.main import svmMLiA


def main_test():
    '''
    主测试函数
    :return:
    '''
    dataArr, labelArr = svmMLiA.loadDataSet(os.path.dirname(os.getcwd()) +
                                            "\\datas\\testSet.txt")
    print "dataArr >>> ", dataArr
    print "labelArr >>> ", labelArr

    # 测试简化版 SMO算法
    b, alpha = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print "b >>> ", b
    print "alpha >>> ", alpha[alpha > 0]

    # 测试完整版 SMO算法
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    b, alpha = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)
    print "b >>> ", b
    print "alpha >>> ", alpha[alpha > 0]

    ws = svmMLiA.calcWs(alpha, dataArr, labelArr)  # 测试calcWs函数
    print "ws >>> ", ws

    # 测试高斯径向基核函数
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    svmMLiA.testRbf()

    # 测试基于SVM的手写数字识别系统
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    svmMLiA.testDigits(('rbf', 20))
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    svmMLiA.testDigits()  # 去参数到达率为10时， 效果最好


if __name__ == '__main__':
    main_test()
