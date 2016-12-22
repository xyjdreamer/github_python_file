#! usr/bin/env python
#-*- coding:utf-8 -*-

#多项式模型
import numpy as np
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocaList(dataSet):   #创建不重复的词列表
    vocaList = set([])
    for document in dataSet:
        vocaList = vocaList | set(document)
    return list(vocaList)

def setWord2Vector(wordList,inputSet):  #为输入数据创建词向量
    wordVector = [0]*len(wordList)
    for word in inputSet:
        if word in wordList:
            wordVector[wordList.index(word)] += 1
    return wordVector

#此处假设为二元分类问题，标签值只有0,1之分
def trainNB0(trainSet,trainLabels,V):
#先验概率P(c) = 类c下单词总数/整个训练样本的单词总数
#类条件概率P(tk|c) = (类c下单词tk在各个文档中出现过的次数之和+1)/(类c下单词总数+|V|)
    p1Num = np.ones(len(trainSet[0]))
    p0Num = np.ones(len(trainSet[0]))
    for i in range(len(trainLabels)):
        if trainLabels[i] == 1:
            p1Num += trainSet[i]
        else:
            p0Num += trainSet[i]
    p1Denom = np.sum(p1Num)
    p0Denom = np.sum(p0Num)
    pc1 = p1Denom/float(p1Denom+p0Denom)
    p1 = p1Num/(p1Denom+V)
    p0 = p0Num/(p0Denom+V)
    return p1 , p0, pc1

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass):
    p1 = np.sum(vec2Classify*np.log(p1Vec)) + np.log(pClass)
    p0 = np.sum(vec2Classify*np.log(p0Vec)) + np.log(1-pClass)
    if p1 > p0:
        return 1   #1代表垃圾邮件,p1代表为垃圾邮件的概率
    else:
        return 0   #0代表非垃圾邮件，p0代表非垃圾邮件的概率

def testNB():
    import re
    import os
    import random
    hamFileName = os.listdir('email\\ham')
    spamFileName = os.listdir('email\\spam')
    Data = []   #每个邮件的单词组成的列表
    classVec = []   #每个邮件的标签
    for name in hamFileName:   #依次读取非垃圾邮件文件
        fp = open('email\\ham\\'+name)
        str = fp.read()
        Data.append(re.split(re.compile('\W*'),str))
        classVec.append(0)  #设置 0 代表非垃圾邮件
    for name in spamFileName:  #依次读取垃圾邮件文件
        fp = open('email\\spam\\'+name)
        str = fp.read()
        Data.append(re.split(re.compile('\W*'),str))
        classVec.append(1) #设置 1 代表垃圾邮件

    #选择测试集，并从全部的训练集中扣除相应的测试例
    testData = []
    testDataClassVec = []
    trainData = Data
    trainDataClassVec = classVec

    #随机生成10个用于测试的数据的索引
    for i in range(10):
        index = random.randrange(0,len(Data))
        testData.append(Data[index])
        testDataClassVec.append(classVec[index])
        del Data[index],classVec[index]


    wordList = createVocaList(trainData)  #为数据集构建单词列表
    wordVector = []
    for i in range(len(trainData)):
        oneEmail = trainData[i]
        wordVector.append(setWord2Vector(wordList,oneEmail))

    p1, p0, pc1 = trainNB0(wordVector,trainDataClassVec,len(wordList))

    rightCount = 0
    for i in range(len(testData)):
        oneEmail = testData[i]
        word2Vector = setWord2Vector(wordList,oneEmail)
        predictResult = classifyNB(word2Vector,p0,p1,pc1)
        if predictResult == testDataClassVec[i]:
            rightCount += 1
    print 'right rate is %.3f%%'%(float(rightCount)/len(testData)*100.0)


testNB()
