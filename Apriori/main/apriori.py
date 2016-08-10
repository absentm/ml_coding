# coding=utf-8
# apriori.py
# Apriori算法

from time import sleep
from votesmart import votesmart

votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
from numpy import *
import os


def loadDataSet():
    '''
    简单数据集，用于测试
    :return:
    '''
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    '''
    构建集合 C1
    :param dataSet: 数据集
    :return:
    '''
    C1 = []

    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    # 对C1中的每个项构建一个不变集合, frozenset: Python自带关键字，控制不可变形
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    '''
    从C1生成L1
    :param D: 包含候选集合的列表
    :param Ck: 数据集
    :param minSupport: 感兴趣项集的最小支持度
    :return:
    '''
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = float(len(D))
    retList = []
    supportData = {}

    for key in ssCnt:
        support = ssCnt[key] / numItems  # 计算所有项集的支持度
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support

    return retList, supportData


def aprioriGen(Lk, k):
    '''
    创建候选集 Ck
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    :return:
    '''
    retList = []
    lenLk = len(Lk)

    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 如果前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]

            L1.sort()
            L2.sort()

            if L1 == L2:
                retList.append(Lk[i] | Lk[j])  # set union

    return retList


def apriori(dataSet, minSupport=0.5):
    '''
    apriori算法主程序
    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return:
    '''
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2

    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)  # 扫描数据集，从Ck得到Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1

    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    '''
    生成关联规则
    :param L: 频繁项集列表
    :param supportData: 包含频繁项集支持数据的字典
    :param minConf: 最小可信度阈值
    :return: 规则列表
    '''
    bigRuleList = []  # 存储规则列表

    for i in range(1, len(L)):  # 只获取两个或更多元素的集合
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]

            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)

    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    '''
    对规则进行评估，计算可信度
    :param freqSet:
    :param H:
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    '''
    prunedH = []

    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算可信度

        if conf >= minConf:
            print freqSet - conseq, '-->', conseq, 'conf:', conf
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)

    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
    生成候选集合
    :param freqSet:
    :param H:
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    '''
    m = len(H[0])

    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)

        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def getActionIds():
    '''
    收集美国国会议案中的Action ID
    :return: actionIdList, Action ID; billTitleList, 标题
    '''
    actionIdList = []
    billTitleList = []

    fr = open(os.path.dirname(os.getcwd()) +
              '\\datas\\recent20bills.txt')

    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)

            for action in billDetail.actions:
                # 过滤掉包含投票的行为
                if action.level == 'House' and \
                        (action.stage == 'Passage' or
                                 action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)

    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList):
    '''
    基于投票数据的事务列表填充
    :param actionIdList:
    :param billTitleList:
    :return:
    '''
    itemMeaning = ['Republican', 'Democratic']

    for billTitle in billTitleList:
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)

    transDict = {}
    voteCount = 2

    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print "problem getting actionId: %d" % actionId
        voteCount += 2

    return transDict, itemMeaning
