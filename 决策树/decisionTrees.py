#!usr/bin/enc python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#计算指定数据集的熵（即计算该数据集的纯度，另一种计算纯度的方法为基尼系数）
def clacEntropy(dataSet):
    m = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        label = featVec[-1]
        labelCount[label] = labelCount.get(label,0) + 1
    Entorpy = 0.0
    for key in labelCount.keys():
        prob = float(labelCount[key])/m
        Entorpy += (-prob*np.log2(prob))
    return Entorpy

#划分数据，dataSet为待划分的数据集，axis代表划分依据的特征的索引，value代表划分依据的特征的值
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for data in dataSet:
        if data[axis] == value:
            a = data[:axis]
            a.extend(data[axis+1:])
            retDataSet.append(a)
    return retDataSet


#选择最佳的划分方式,
def chooseBestSplit(DataSet):
    numFeature = len(DataSet[0])-1  #原始数据的特征数，最后一个代表标签值
    bastEntropy = clacEntropy(DataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        newEntropy = 0.0
        featureList = set([example[i] for example in DataSet]) #获取第i个特征的全部值,并且去重复，类型为列表
        for value in featureList:
            subDataSet = splitDataSet(DataSet,i,value)
            prob = len(subDataSet)/float(len(DataSet))
            newEntropy += prob*clacEntropy(subDataSet)
        infoGain = bastEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for value in classList:
        classCount[value] = classCount.get(value,0) + 1
    sortedCounted = sorted(classCount.iteritems(),lambda dic : dic[1],reverse=True)
    return sortedCounted[0][0]

#originDataSet的最后一列即代表了该条数据的标签
def createTree(originDataSet,featureNames):
    labelsList = [example[-1] for example in originDataSet] #此处labelList的类型为列表
    if labelsList.count(labelsList[0]) == len(labelsList): #如果只有一个类，直接返回该类的标签值
        return labelsList[0]
    if len(originDataSet[0]) == 1:  #如果使用完了标签的所有特征
        return majorityCnt(labelsList)

    bestFeature = chooseBestSplit(originDataSet)
    bestFeatureName = featureNames[bestFeature]
    myTree = {bestFeatureName : {}}
    del featureNames[bestFeature]
    featureValues = set([example[bestFeature] for example in originDataSet])
    for value in featureValues:
        subFeatureName = featureNames[:]
        myTree[bestFeatureName][value] = createTree(splitDataSet(originDataSet,bestFeature,value),subFeatureName)
    return myTree

#测试集是以列表的方式输入，不是以数组的方式输入
dataSet = [
           [1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'],
           [0,1,'no'],
           [0,1,'no']
           ]
labels = ['no surfacing','flipper']
print createTree(dataSet,labels)

