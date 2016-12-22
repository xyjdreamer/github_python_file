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
    def train(self,input,label,learning_rate=0.1,epochs=5000):
        #增加偏置
        input_bias = np.hstack((input,np.ones((input.shape[0],1))))
        #初始化权值,列向量
        self.weight = 2 * np.random.random((input_bias.shape[1],1)) - 1
        iter = 0
        flag = True
        while iter < epochs and flag:
            iter += 1
            flag = False
            for i in range(0,input_bias.shape[0]):  #遍历所有样本
                X = label[i]*input_bias[i]
                #预测错误的情况：
                #      实际值  预测值
                #        +1     -1
                #        -1     +1
                #故如果预测有误，则相乘结果小于0
                #权值更新规则：
                #如果实际标签为+1   w = w + learning_rate*input
                #如果实际标签为-1   w = w - learning_rate*input
                if np.dot(X,self.weight) <= 0:  #如果预测错误
                    flag = True
                    self.weight = self.weight + learning_rate*X.reshape((-1,1))

    def predict(self,data):
        data_bias = np.hstack((data,np.ones(1)))
        temp = np.dot(data_bias,self.weight)
        if temp>0: temp = 1
        else:      temp = 0
        return temp

if __name__=='__main__':

    print 'OR Problem:'
    #OR
    data1 = np.array([[0,0],[0,1],[1,0],[1,1]])
    data2 = np.array([-1,1,1,1])

    nn_or = perceptron()
    nn_or.train(data1,data2)
    for test in data1:
        print test,'--->',nn_or.predict(test)

    print 'AND Problem:'
    #AND
    data2 = np.array([-1, -1, -1, 1])

    nn_and = perceptron()
    nn_and.train(data1, data2)
    for test in data1:
        print test, '--->',nn_and.predict(test)














            