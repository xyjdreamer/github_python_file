#!usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt',delimiter=',')
x = data[:,0]
y = data[:,1]

plt.figure()
plt.plot(x,y,'rD')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.title('Scatter plot of training data')
plt.show()

