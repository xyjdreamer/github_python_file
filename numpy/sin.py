#!usr/bin/env python
#-*- coding:utf-8 -*-
import math
import time
import numpy as np

x = [i*0.001 for i in xrange(10000000)]

start = time.clock()
for i,t in enumerate(x):
    math.sin(t)
end = time.clock()
print "math.sin:",end-start

start = time.clock()
np.sin(x)  #np.sin支持对数组进行操作，而math.sin仅仅支持对单个数进行操作
end = time.clock()
print "np.sin:",end-start
