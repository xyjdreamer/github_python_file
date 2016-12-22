#!usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#计算J(θ)
def computerCost(var_X,var_y,var_theta):
    i = var_X*var_theta - var_y.reshape(-1,1)
    J = (i.T)*i/(len(var_y)*2)
    return J
#利用批梯度下降法迭代求θ
def gradientDescent(var_X,var_y,var_alpha,var_theta,var_iterations):
    J_history = np.empty(var_iterations)    #记录每一次迭代 J(θ)的值
    for i in range(var_iterations):
        var_theta = var_theta + var_alpha*(var_X.T)*(var_y.reshape(-1,1) - var_X*var_theta)/len(var_y)
        J_history[i] = computerCost(var_X,var_y,var_theta)
    return var_theta , J_history

#绘制训练集的散点图
data = np.loadtxt('ex1data1.txt',delimiter=',')
x1 = data[:,0]
y = data[:,1]

plt.figure()
plt.plot(x1,y,'rD',label='examples')

#Gradient Descent
m = len(y)   #训练数目
X = np.matrix(np.column_stack([np.ones((m,1)),x1.reshape(-1,1)]))
theta = np.matlib.zeros((2,1))   #初始化θ为0向量
iterations = 1500   #迭代次数
alpha = 0.01    #初始化α

theta , J = gradientDescent(X,y,alpha,theta,iterations)

x = np.linspace(5,25,1000)
m = np.matrix(np.column_stack([np.ones((1000,1)),x.reshape(-1,1)]))
h = m * theta
plt.plot(x,h,label='predict')

plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.title('Scatter plot of training data & Predict')
plt.legend(loc='best')  #参数指定图示的位置
plt.show()

#show J(θ)
theta0 = np.linspace(-10,10,100)
theta1 = np.linspace(-1,4,100)
J_vals = np.empty((len(theta0),len(theta1)))

for i in range(len(theta0)):
    for j in range(len(theta1)):
        t = np.matrix([theta0[i] , theta1[j]]).T
        J_vals[i,j] = computerCost(X,y,t)

fig = plt.figure()
ax = fig.gca(projection='3d')

theta0 , theta1 = np.meshgrid(theta0,theta1)
ax.plot_surface(theta0,theta1, J_vals.T, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.show()