# coding=utf-8
# fpGrowth.py
# FP-growth算法

import twitter
from time import sleep
import re


class treeNode:
    '''
    FP树结构
    '''

    def __init__(self, nameValue, numOccur, parentNode):
        '''
        构造函数，FP树的属性
        :param nameValue: 存放节点名字的变量
        :param numOccur: 计数值
        :param parentNode: 指向当前节点的父节点
        :return:
        '''
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        '''
        自增计数
        :param numOccur:
        :return:
        '''
        self.count += numOccur

    def disp(self, ind=1):
        '''
        树结构的显示
        :param ind:
        :return:
        '''
        print '  ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):
    '''
    构建FP树
    :param dataSet: 数据集
    :param minSup: 最小支持度，可选
    :return:
    '''
    headerTable = {}

    # 遍历数据集两次
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    # 移除不满足最小支持度的元素项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del (headerTable[k])
    freqItemSet = set(headerTable.keys())

    if len(freqItemSet) == 0:
        return None, None  # 如果没有元素项满足要求退出

    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    print 'headerTable: ', headerTable
    retTree = treeNode('Null Set', 1, None)

    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    '''
    更新FP树
    :param items:
    :param inTree:
    :param headerTable:
    :param count:
    :return:
    '''
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:  # 添加items[0]
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    '''
    更新头结点，保证节点链接指向树中该元素项的每一个实例
    :param nodeToTest: 测试节点
    :param targetNode: 目标节点
    :return:
    '''
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink

    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    '''
    简单数据集的构建
    :return:
    '''
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

    return simpDat


def createInitSet(dataSet):
    '''
    将数据转换成字典类型
    :param dataSet: 数据集
    :return:
    '''
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1

    return retDict


def ascendTree(leafNode, prefixPath):
    '''
    上溯FP树
    :param leafNode: 叶节点
    :param prefixPath: 条件模式基
    :return:
    '''
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    '''
    生成条件模式基字典
    :param basePat:
    :param treeNode:
    :return:
    '''
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)

        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink

    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    '''
    递归查找频繁项集
    :param inTree:
    :param headerTable:
    :param minSup:
    :param preFix:
    :param freqItemList:
    :return:
    '''
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]

    for basePat in bigL:  # 从头指针表的底端开始
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)

        if myHead != None:
            print 'conditional tree for: ', newFreqSet
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*',
                         '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)

    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    # you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1, 15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList
