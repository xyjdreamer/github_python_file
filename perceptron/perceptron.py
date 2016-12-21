#!usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

'''
单层感知器
'''
class perceptron():
    def __init__(self):
        pass
    #label = -1, +1
    def train(self,input,label,learning_rate=0.1,epochs=500):
        #增加偏置
        input_bias = np.hstack((input,np.ones((input.shape[0],1))))
        #初始化权值,列向量
        self.weight = 2 * np.random.random((input_bias.shape[1],1)) - 1
        iter = 0
        p = np.dot(input_bias,self.weight)
        temp = label*p.reshape((1,-1))
        temp[temp>0] = 0
        temp[temp<=0] = 1
        while iter < epochs and np.sum(temp)>0:
            iter += 1
            index = np.random.randint(input_bias.shape[0])
            X = input_bias[index]
            Y = np.dot(X,self.weight)
            if label[index]*Y <= 0: # 如果预测结果有错,更新权值
                self.weight  = self.weight + learning_rate*Y*label[index]

            p = np.dot(input_bias, self.weight)
            temp = label * p.reshape((1, -1))
            temp[temp > 0] = 0
            temp[temp <= 0] = 1
        print temp

    def predict(self,data):
        data_bias = np.hstack((data,np.ones(1)))
        return np.dot(data_bias,self.weight)

if __name__=='__main__':
    data1 = np.array([[0,0],[0,1],[1,0],[1,1]])
    data2 = np.array([-1,1,1,1])

    nn = perceptron()
    nn.train(data1,data2)
    for test in data1:
        print nn.predict(test)













            