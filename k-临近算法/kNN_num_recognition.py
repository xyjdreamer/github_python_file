#!usr/bin/env python
#-*- coding: utf-8 -*-

import os
import numpy as np
import numpy.linalg
import time
import matplotlib.pyplot as plt

def img2vector(filename):
    fp = open(filename,'r')
    returnVector = np.zeros((1,1024))  #将32*32的图像转化为一维数组
    for i in range(32):
        str = fp.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(str[j])
    return returnVector
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

def handwritingClassTest():
    #从文件创建dataSet,及其相应的labels
    trainingfileSet = os.listdir('digits\\trainingDigits')
    traingingDataLabels = np.zeros(len(trainingfileSet))
    dataSet = np.zeros((len(trainingfileSet),1024))
    for i in range(len(trainingfileSet)):
        str = trainingfileSet[i]   #获取文件名
        traingingDataLabels[i] = int(str.split('_')[0])
        dataSet[i] = img2vector('digits\\trainingDigits'+'\\'+str)

    #利用测试集对算法进行测试
    testfileSet = os.listdir('digits\\testDigits')
    errorCount = 0
    for i in range(len(testfileSet)):
        testVector = img2vector('digits\\testDigits'+'\\'+testfileSet[i])
        label = int(testfileSet[i].split('_')[0])
        predict = kClassify(testVector, dataSet, traingingDataLabels, k=9)
        #print 'the predict num is:',predict,'---->the real num is:',label
        if predict != label:
            errorCount += 1
    print '训练集大小为：',len(traingingDataLabels)
    print '总测试数为：',len(testfileSet),'预测错误数为：',errorCount
    print '错误率为%.2f%%' %  (errorCount/float(len(testfileSet))*100)


start = time.clock()
handwritingClassTest()
print '程序耗时 %.3fs'% (time.clock()-start)