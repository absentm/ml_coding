## 主成分分析：PCA

* 降维流程：

	对于数据需要提取的第一个主成分是从数据差异性最大（即方差最大）方向，第二个主成分则来自于数据差异性次大的方向，并且该方向与第一个主成分方向正交。

	通过数据集的协方差矩阵及其特征值分析，就可以求得这些主成分的值。
		
	得到协方差矩阵的特征向量，保留最大的N个值，将数据乘上这N个特征向量从而将其转化到新的空间。
	


----------


* 算法伪代码实现：[将数据转化为前N个主成分流程]
		
	1. 去除平均值
	2. 计算协方差矩阵
	3. 计算协方差矩阵的特征值和特征向量
	4. 将特征值从大到小排序
	5. 保留最上面的N个特征向量
	6. 将数据转化到上述N个特征向量构建的新空间中
	7. 


----------

* Python实现

	    def pca(dataMat, topNfeat=9999999):
		    '''
		    PCA算法的实现
		    :param dataMat: 数据集
		    :param topNfeat: 应用的N个特征，可选
		    :return: lowDDataMat, 降维后的数据集； reconMat，被重构后的原始数据集
		    '''
		    meanVals = mean(dataMat, axis=0)  # 计算均值
		    meanRemoved = dataMat - meanVals  # 去除均值
			covMat = cov(meanRemoved, rowvar=0)  # 计算协方差
		    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算协方差矩阵的特征值和特征向量

		    eigValInd = argsort(eigVals)  # 排序，从小到大
		    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 去除多余的维度
		    redEigVects = eigVects[:, eigValInd]  # 重组特征向量从大到小
		    lowDDataMat = meanRemoved * redEigVects  # 将数据转化到新的空间
		    reconMat = (lowDDataMat * redEigVects.T) + meanVals
	    
		    return lowDDataMat, reconMat