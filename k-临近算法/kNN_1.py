#!usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

#k-临近算法
#计算预测点与已知点的欧几里得距离，然后取前 K个距离最小的点，统计这些点各自的分类，选取频率出现最高的类作为预测结果
def kClassify(inX,dataSet,labels,k):
    diffMatrix = np.tile(inX,(np.size(dataSet,0),1)) - dataSet
    dist = numpy.linalg.norm(diffMatrix,axis=1)   #计算未知数据与各个点的欧几里得距离
    sortedDistanceIndex = np.argsort(dist)  #排序，返回从小到大值的索引值
    classCount = {}   #空字典
    for i in range(k):
        voteLabels = labels[sortedDistanceIndex[i]]   #获取相应索引的标签值
        classCount[voteLabels] = classCount.get(voteLabels,0) + 1 #字典的键值为标签，键对应的值为相应的统计计数
    #对字典按照值排序
    sortedClassCount = sorted(classCount.iteritems(),key=lambda dit : dit[1],reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    data = np.loadtxt(filename,delimiter='\t')  #读取txt文本数据
    featureMatrix = data[:,0:3]
    labelsVector = data[:,3]
    return featureMatrix,labelsVector

def normalFeature(dataSet):  #归一化，将每一个特征的各个值都归一化到0~1之间
    minVals = np.min(dataSet,axis=0)
    maxVals = np.max(dataSet,axis=0)
    newDataset = (dataSet - np.tile(minVals,(np.size(dataSet,0),1)))/(np.tile(maxVals-minVals,(np.size(dataSet,0),1)))
    return  newDataset

#测试分类器的准确率
def dateClassTest():
    featureMatrix,labelsVector = file2matrix('datingTestSet2.txt')  #读取数据
    norm_featureMtrix = normalFeature(featureMatrix) #归一化
    testRate = 0.1   #设定测试率
    m = np.size(norm_featureMtrix,0)  #总样本数
    testNum = m*testRate  #测试样本数
    errorCount = 0
    for i in range(int(testNum)):
        result = kClassify(norm_featureMtrix[i],norm_featureMtrix[testNum:],labelsVector[testNum:],5)
        if result != labelsVector[i]:
            errorCount += 1
        print 'predict value is :',result,'------real value is :',labelsVector[i]
    print "error rate is %.2f%% "% (errorCount/float(testNum)*100)


dateClassTest()