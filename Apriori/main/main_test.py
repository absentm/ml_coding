# coding=utf-8
# main_test.py
# 主测试函数

import os

from Apriori.main import apriori


def simple_test():
    '''
    简单数据集上的测试
    :return:
    '''
    dataSet = apriori.loadDataSet()
    print "dataSet >>> ", dataSet

    C1 = apriori.createC1(dataSet)
    print "C1 >>> ", C1

    D = map(set, dataSet)
    print "D >>> ", D

    L1, suppData0 = apriori.scanD(D, C1, 0.5)
    print "L1 >>> ", L1
    print "suppData0 >>> ", suppData0

    L, suppData = apriori.apriori(dataSet)
    print "L >>> ", L
    print "suppData >>> ", suppData

    L, suppData = apriori.apriori(dataSet, minSupport=0.5)
    print "L >>> ", L
    print "suppData >>> ", suppData

    rules = apriori.generateRules(L, suppData, minConf=0.7)
    print "rules >>> ", rules


def voteTest():
    '''
    美国国会选举投票数据集测试
    :return:
    '''
    actionIdList, billTitles = apriori.getActionIds()
    print "actionIdList ", actionIdList
    print "billTitles ", billTitles

    transDict, itemMeaning = apriori.getTransList(actionIdList, billTitles)
    print "transDict ", transDict
    print "itemMeaning ", itemMeaning

    dataSet = [transDict[key] for key in transDict.keys()]
    L, suppData = apriori.apriori(dataSet, minSupport=0.7)
    rules = apriori.generateRules(L, suppData, minConf=0.95)
    print rules


def mushroomTest():
    '''
    毒蘑菇数据集的测试
    :return:
    '''
    mushDataSet = [line.split() for line
                   in open(os.path.dirname(os.getcwd()) +
                           "\\datas\\mushroom.dat").readlines()]
    L, suppData = apriori.apriori(mushDataSet, minSupport=0.3)

    for item in L[1]:
        if item.intersection('2'):
            print item


def main_test():
    print "apriori"
    simple_test()
    voteTest()
    mushroomTest()


if __name__ == '__main__':
    main_test()
