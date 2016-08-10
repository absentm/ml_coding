# coding=utf-8
# main_test.py
# 主测试程序
from numpy import mat
from numpy.linalg import linalg

from SVD.main import svd


def simple_test():
    '''
    简单数据集测试
    :return:
    '''
    data = svd.loadExData()
    U, Sigma, VT = linalg.svd(data)
    print "U >>> ", U
    print "Sigma >>> ", Sigma
    print "VT >>> ", VT

    # 重构原矩阵
    sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    print "U[:, :3] * sig3 * VT[:3, :] >>> ", U[:, :3] * sig3 * VT[:3, :]

    myMat = mat(data)
    ecludSim = svd.ecludSim(myMat[:, 0], myMat[:, 4])
    pearSim = svd.pearsSim(myMat[:, 0], myMat[:, 4])
    cosSim = svd.cosSim(myMat[:, 0], myMat[:, 4])

    print "ecludSim >>> ", ecludSim
    print "pearSim >>> ", pearSim
    print "cosSim >>> ", cosSim

    myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    myMat[3, 3] = 2
    print "myMat >>> ", myMat
    print "svd.recommend(myMat, 2) >>> ", svd.recommend(myMat, 2)
    print "svd.recommend(myMat, 2, ) >>> ", svd.recommend(myMat, 2, simMeas=svd.ecludSim)
    print "svd.recommend(myMat, 1, ..) >>> ", svd.recommend(myMat, 1, estMethod=svd.svdEst, simMeas=svd.pearsSim)


    svd.imgCompress(2)

def main_test():
    print "SVD"
    simple_test()


if __name__ == '__main__':
    main_test()
