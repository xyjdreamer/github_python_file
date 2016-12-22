#!usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,1000) #0-10之间生成1000个数
y = np.sin(x)  #利用np.sin计算数组x的sin值
z = np.cos(x)  #利用np.cos计算数组x的cos值

plt.figure()
plt.plot(x,y,'r-',label='$sin(x)$',linewidth=2)
plt.plot(x,z,'b--',label='$cos(x)$',linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Pyplot First Example")
plt.legend()
plt.ylim(-1.5,1.5)
plt.show()