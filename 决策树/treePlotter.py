#!usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#获取叶子节点数，tree以字典的方式存储
def getleafNum(tree):
    leafNum = 0
    firstNode = tree.keys()[0]
    secondNode = tree[firstNode]
    for key in secondNode.keys():
        if type(secondNode[key]).__name__ == 'dict':
            leafNum += getleafNum(secondNode[key])
        else:
            leafNum += 1
    return  leafNum
def getTreeDepth(tree):
    maxdepth = 0
    curdepth = 0
    firstNode = tree.keys()[0]
    secondNode = tree[firstNode]
    for key in secondNode.keys():
        if type(secondNode[key]).__name__ == 'dict':
            curdepth = 1 + getTreeDepth(secondNode[key])
        else:
            curdepth = 1
        if curdepth > maxdepth : maxdepth = curdepth
    return maxdepth

#parentPt 代表需要绘制的点的坐标，centerPt 代表注释的坐标
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

#tree代表需要绘制的树,parentPtr代表根节点的坐标位置,nodeTxt代表
def plotTree(tree,parentPtr,nodeTxt):
    numLeafs = getleafNum(tree)
    depth = getTreeDepth(tree)
    firstStr = tree.keys()[0]
    centerPt = (plotTree.xOff + (1.0+float(numLeafs))/2.0/plotTree.totalW , plotTree.yOff)
    plotMidText(centerPt,parentPtr,nodeTxt)
    plotNode(firstStr,centerPt,parentPtr,decisionNode)
    secondDict = tree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],centerPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),centerPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),centerPt,str(key))
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getleafNum(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

createPlot(retrieveTree(0))




