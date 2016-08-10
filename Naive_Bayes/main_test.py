# coding=utf-8
# main_test.py
# 主程序测试

import feedparser

from Naive_Bayes import bayes


def main_test():
    # 加载文档数据
    listPosts, listClass = bayes.loadDataSet()
    print "listPost >>> ", listPosts
    print "listClass >>> ", listClass

    # 生成无序不重复文档集合
    myVocabList = bayes.createVocabList(listPosts)
    print "myVocabList >>> ", myVocabList

    # 测试setOfWords2Vec函数
    print "setOfWords2Vec0 >>> ", bayes.setOfWords2Vec(myVocabList, listPosts[0])
    print "setOfWords2Vec1 >>> ", bayes.setOfWords2Vec(myVocabList, listPosts[3])

    # 测试trainNB0函数
    trainMat = []
    for postInDoc in listPosts:
        trainMat.append(bayes.setOfWords2Vec(myVocabList, postInDoc))
    p0V, p1V, pAb = bayes.trainNB0(trainMat, listClass)
    print "p0V >>> ", p0V
    print "p1V >>> ", p1V
    print "pAb >>> ", pAb  # 任意文档属于侮辱性文档的概率

    # 封装测试代码
    bayes.testingNB()

    # 测试垃圾邮件分类系统
    bayes.spamTest()

    # 测试RSS源分类器
    print "----------------------------------------------------------"
    ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
    sy = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")
    vocabList, pSF, pNY = bayes.localWords(ny, sy)
    print "vocabList >>> ", vocabList
    print "pSF >>> ", pSF
    print "pNY >>> ", pNY
    bayes.getTopWords(ny, sy)


if __name__ == '__main__':
    main_test()
