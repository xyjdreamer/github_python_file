#! usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import numpy.linalg
from matplotlib import cm
from  mpl_toolkits.mplot3d import Axes3D
import copy
#由于X的两个特征的值相差较大，第一个特征可能会对预测结果产生较大影响，因此需要规范化X，求X各个特征值的mean和standard deviation
#修改后的X的各个特征值的mean=0，standard deviation=1,
#新的X求解公式为（X-mean）/standard deviation
def featureNormalization(var_X):
    mean = np.zeros(np.size(var_X,1))   #mean定义为一维数组即可
    sigma = np.zeros(np.size(var_X,1))
    norm_X = var_X
    for i in range(len(mean)):
        mean[i] =  np.mean(var_X[:,i])
        sigma[i] = np.std(var_X[:,i])
        norm_X[:,i] = (norm_X[:,i] - mean[i])/sigma[i]
    return norm_X,mean,sigma

#计算J(θ)
def computeCostMulti(var_X,var_y,var_theta):
    i = var_X*var_theta - var_y
    J = (i.T)*i/(2*len(var_y))
    return J

#batch Gradient Descent
def gradientDescentMulti(var_X,var_y,var_theta,var_alpha,var_iterations):
    J_history = np.empty(var_iterations)
    for i in range(var_iterations):
        var_theta = var_theta + var_alpha*(var_X.T)*(var_y-var_X*var_theta)/len(var_y)
        J_history[i] = computeCostMulti(var_X,var_y,var_theta)
    return var_theta , J_history

#Normal equation
def normalEquation(var_X,var_y):
    #return numpy.linalg.inv(var_X.T*var_X)*var_X.T*var_y
    return (var_X.T*var_X).I*var_X.T*var_y


data = np.loadtxt("ex1data2.txt",delimiter=',')
xs = copy.deepcopy(data[:,0])
ys = copy.deepcopy(data[:,1])
zs = copy.deepcopy(data[:,2])
X = data[:,0:2]     #默认进行浅拷贝
y = (data[:,2].reshape(-1,1))/10000  #除以10000，将房屋价格按照万元计算
norm_X,mean,sigma =featureNormalization(X)
#添加x0 = 1
norm_X = np.matrix(np.column_stack([np.ones((np.size(norm_X,0),1)),norm_X]))
theta = np.matlib.zeros((np.size(norm_X,1),1))
alpha = 0.1
iterations = 100

theta1 , J = gradientDescentMulti(norm_X,y,theta,alpha,iterations)
theta2 = normalEquation(norm_X,y)
print 'theta2',theta2
print 'theta1',theta1
np.set_printoptions(precision=3,suppress=True,threshold=20000)      #设置数据打印的格式，此处设置为不以科学计数法形式输出

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xs,ys,zs,color='r',marker='o')
ax.set_xlabel('size')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price*10000$')
plt.show()

#
alist = [0.01,0.05,0.1,0.2,0.3]
color = ['r','b','g','k','y']
iterations_num = np.array(range(iterations))
plt.figure()
for i in range(len(alist)):
    theta ,J = gradientDescentMulti(norm_X, y, np.matlib.zeros((np.size(norm_X,1),1)), alist[i], iterations)
    plt.plot(iterations_num,J,color=color[i],label=('alpha=%.2f'%alist[i]))
plt.legend(loc='best')
plt.show()