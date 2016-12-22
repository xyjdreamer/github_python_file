#!usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import yappi

def readDataFromTxt(filename):
    data = np.loadtxt(filename,delimiter='\t')
    featureArray = np.column_stack([np.ones((np.size(data,0),1)),data[:,0:2]])
    labelArray = data[:,-1]
    return featureArray , labelArray

def logisticFunction(x):
    return 1/(1 + np.exp(-x))

def batchGradientAscentMulti(var_X,var_y):
    theta = np.zeros((np.size(var_X,1),1))#构建一维列向量
    featureMatrix = np.matlib.mat(var_X)
    labelMatrix = np.matlib.mat(var_y)
    iterations = 500
    alpha = 0.001
    for i in range(iterations):
        h = logisticFunction(featureMatrix*theta)#此处var_X和theta为矩阵相乘
        theta = theta + alpha*(featureMatrix.T)*(labelMatrix - h)
    return theta

def stochasticGradientAscentMulti(var_X,var_y,iterations = 150):
    theta = np.zeros(np.size(var_X,1)) #构建一维列向量
    theta0_history = np.zeros(iterations * np.size(var_X, 0))
    theta1_history = np.zeros(iterations * np.size(var_X, 0))
    theta2_history = np.zeros(iterations * np.size(var_X, 0))
    alpha = 0.001
    for j in range(iterations):
        for i in range(len(var_X)):
            h = logisticFunction(np.sum(var_X[i]*theta))  #此处var_X和theta为向量相乘，与矩阵相乘不同
            theta = theta + alpha*(var_y[i] - h)*var_X[i]
            theta0_history[j * np.size(var_X, 0) + i] = theta[0]
            theta1_history[j * np.size(var_X, 0) + i] = theta[1]
            theta2_history[j * np.size(var_X, 0) + i] = theta[2]
    return theta,theta0_history,theta1_history,theta2_history

def stochasticGradientAscentMulti_1(var_X,var_y,iterations = 150):
    import random
    theta = np.zeros(np.size(var_X,1)) #构建一维列向量
    theta0_history = np.zeros(iterations * np.size(var_X, 0))
    theta1_history = np.zeros(iterations * np.size(var_X, 0))
    theta2_history = np.zeros(iterations * np.size(var_X, 0))
    for j in range(iterations):
        indexList = range(len(var_X))
        random.shuffle(indexList)   #将训练集随机分布
        for k , i in enumerate(indexList):
            alpha = 4/(1.0+j+i)+0.1  #每次更新alpha
            h = logisticFunction(np.sum(var_X[i]*theta))  #此处var_X和theta为向量相乘，与矩阵相乘不同
            theta = theta + alpha*(var_y[i] - h)*var_X[i]
            theta0_history[j * np.size(var_X, 0) + k] = theta[0]
            theta1_history[j * np.size(var_X, 0) + k] = theta[1]
            theta2_history[j * np.size(var_X, 0) + k] = theta[2]
    return theta,theta0_history,theta1_history,theta2_history

np.set_printoptions(suppress=True)
featureArray, labelArray = readDataFromTxt('testSet.txt')
yappi.clear_stats()
yappi.start()
theta = batchGradientAscentMulti(featureArray,labelArray.reshape((-1,1)))
yappi.stop()
stats = yappi.convert2pstats(yappi.get_func_stats())
stats.sort_stats("cumulative")
stats.print_stats()

yappi.clear_stats()
yappi.start()
theta1,theta0_history,theta1_history,theta2_history = \
    stochasticGradientAscentMulti_1(featureArray,labelArray.reshape((-1,1)),40)
yappi.stop()
stats = yappi.convert2pstats(yappi.get_func_stats())
stats.sort_stats("cumulative")
stats.print_stats()

print theta1


#画图:散点图以及拟合直线
xcord1=[]; ycord1=[]
xcord2=[]; ycord2=[]

for i in range(np.size(labelArray)):
    if labelArray[i] == 1:
        xcord1.append(featureArray[i][1])
        ycord1.append(featureArray[i][2])
    else:
        xcord2.append(featureArray[i][1])
        ycord2.append(featureArray[i][2])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord1,ycord1,s=30,c='red',marker='s',label='class1')
ax.scatter(xcord2,ycord2,s=30,c='green',marker='x',label='class2')

x = np.arange(-4.0,4.0,0.1)
y = (-theta[0]-theta[1]*x)/theta[2]
batch_y = []
for i in range(np.size(y)):
    batch_y.append(y[0,i])
ax.plot(x,batch_y,label='$\\theta_1=%.3f,\\theta_2=%.3f$'%(theta[1],theta[2]),color='black',ls = '-',lw=2)

stochastic_y = (-theta1[0]-theta1[1]*x)/theta1[2]
ax.plot(x,stochastic_y,label='$\\theta_1=%.3f,\\theta_2=%.3f$'%(theta1[1],theta1[2]),color='red',ls = '-',lw=2)
ax.set_xlabel('X1');ax.set_ylabel('X2')
plt.legend(loc='best')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(311)
iterations = np.size(theta1_history)
x = range(0,iterations)
y0 = theta0_history
ax.plot(x,y0)
ax.set_xlabel('iterations');ax.set_ylabel('X0')
ax = fig.add_subplot(312)
y1 = theta1_history
ax.plot(x,y1)
ax.set_xlabel('iterations');ax.set_ylabel('X1')
ax = fig.add_subplot(313)
y2 = theta2_history
ax.plot(x,y2)
ax.set_xlabel('iterations');ax.set_ylabel('X2')
plt.show()