# coding=utf-8
# main_test.py
# 主测试程序

import os

from FP_growth.main import fpGrowth


def simple_test():
    '''
    简单数据集上测试
    :return:
    '''
    simpData = fpGrowth.loadSimpDat()
    initSet = fpGrowth.createInitSet(simpData)
    print "initSet >>> ", initSet

    myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)
    myFPtree.disp()

    print "---------------------------------"
    freqItems = []  # 存储所有的频繁集
    fpGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)  # 显示所有的条件树
    print "freqItems >>> ", freqItems


def newsTest():
    '''
    新闻网站点击流数据集测试：30M文件，100万条数据
    :return:
    '''
    parsedData = [line.split() for line in open(
        os.path.dirname(os.getcwd()) + "\\datas\\kosarak.dat").readlines()]

    initSet = fpGrowth.createInitSet(parsedData)

    myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 100000)
    freqItems = []  # 存储所有的频繁集
    fpGrowth.mineTree(myFPtree, myHeaderTab, 100000, set([]), freqItems)


def main_test():
    '''
    主测试程序
    :return:
    '''
    print "FP_growth"
    simple_test()
    newsTest()


if __name__ == '__main__':
    main_test()
