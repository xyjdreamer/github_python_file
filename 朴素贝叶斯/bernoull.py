#! usr/bin/env python
#-*- coding:utf-8 -*-

#伯努利模型

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

def createVocaList(dataSet):   #创建词向量
    vocaList = set([])
    for document in dataSet:
        vocaList = vocaList | set(document)
    return list(vocaList)

def setWord2Vector(wordList,inputSet):
    wordVector = [0]*len(wordList)
    for word in inputSet:
        if word in wordList:
            wordVector[wordList.index(word)] = 1
    return wordVector

#此处假设为二元分类问题，标签值只有0,1之分
def trainNB0(trainSet,trainLabels):
    # P(c) = 类c下文件总数/整个训练样本的文件总数
    # P(tk|c) = (类c下包含某单词tk的文件数+1)/(类c下文件总数+2)
    pc1 = np.sum(trainLabels)/float(len(trainLabels))
    p1Num = np.ones(len(trainSet[0]))
    p0Num = np.ones(len(trainSet[0]))
    for i in range(len(trainLabels)):
        if trainLabels[i] == 1:
            p1Num += trainSet[i]
        else:
            p0Num += trainSet[i]
    p1Denom = np.sum(p1Num) + 2
    p0Denom = np.sum(p0Num) + 2   #Laplace校准
    return p1Num/p1Denom , p0Num/p0Denom , pc1

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass):
    p1 = np.sum(vec2Classify*np.log(p1Vec)) + np.log(pClass)
    p0 = np.sum(vec2Classify*np.log(p0Vec)) + np.log(1-pClass)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    postingList , classVec = loadDataSet()
    wordList = createVocaList(postingList)
    trainWordVectorList = []
    for i in range(len(postingList)):
        oneLine = postingList[i]
        trainWordVectorList.append(setWord2Vector(wordList,oneLine))
    p1 , p0 ,pc1 = trainNB0(trainWordVectorList,classVec)

    testData = ['love','my','dalmation']
    print testData,classifyNB(setWord2Vector(wordList,testData),p0,p1,pc1)
    testData = ['stupid','garbage']
    print testData,classifyNB(setWord2Vector(wordList,testData),p0,p1,pc1)


testingNB()

fp = open('email\\ham\\1.txt')
print fp.read()

