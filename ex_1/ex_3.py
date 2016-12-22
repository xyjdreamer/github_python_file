#! usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

def normalEquation(var_X,var_y):
    #return numpy.linalg.inv(var_X.T*var_X)*var_X.T*var_y
    return (var_X.T*var_X).I*var_X.T*var_y
data = np.array([
    [1,-890],
    [2,-1411],
    [2,-1560],
    [3,-2220],
    [3,-2091],
    [4,-2878],
    [5,-3537],
    [6,-3268],
    [6,-3920],
    [6,-4163],
    [8,-5471],
    [10,-5157]])

X = np.matrix(np.column_stack((np.ones((np.size(data,0),1)),data[:,0])))
y = data[:,1].reshape(-1,1)
print normalEquation(X,y)

