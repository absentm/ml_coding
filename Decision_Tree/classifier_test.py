# coding=utf-8
# classifier_test.py
# UCI隐形眼镜数据集测试分类器效果

from Decision_Tree import trees
from Decision_Tree import treePlotter


def main_test():
    '''
    隐形眼镜主测试函数
    :return:
    '''
    fr = open("lenses.txt")
    lenses = [inst.strip().split("\t") for inst in fr.readlines()]
    lensesLabels = ["age", "perscript", "astigmatic", "tearRate"]
    lensesTree = trees.createTree(lenses, lensesLabels)
    print "lensesTree >>> ", lensesTree
    treePlotter.createPlot(lensesTree)


if __name__ == '__main__':
    main_test()
