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
        self.weight = np.zeros((input_bias.shape[1],1))#2 * np.random.random((input_bias.shape[1],1)) - 1
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
        return iter

    def predict(self,data):
        data_bias = np.hstack((data,np.ones(1)))
        temp = np.dot(data_bias,self.weight)
        if temp>0: temp = 1
        else:      temp = 0
        return temp

def draw(data_1,data_2):
    import matplotlib.pyplot as plt
    data1 = data_1[0:10]
    data2 = data_1[10:20]
    data3 = data_2[0:10]
    data4 = data_2[10:20]
    plt.figure()
    plt.subplot(121)
    plt.scatter(data1[:,0],data1[:,1],marker='o',c='r',label='$w1$')
    plt.scatter(data2[:,0],data2[:,1],marker='s',c='g',label='$w2$')
    plt.legend()
    plt.subplot(122)
    plt.scatter(data3[:, 0], data3[:, 1], marker='o', c='r', label='$w1$')
    plt.scatter(data4[:, 0], data4[:, 1], marker='s', c='g', label='$w2$')
    plt.legend()
    plt.show()

if __name__=='__main__':

    # print 'OR Problem:'
    # #OR
    # data1 = np.array([[0,0],[0,1],[1,0],[1,1]])
    # data2 = np.array([-1,1,1,1])
    #
    # nn_or = perceptron()
    # nn_or.train(data1,data2)
    # for test in data1:
    #     print test,'--->',nn_or.predict(test)
    #
    # print 'AND Problem:'
    # #AND
    # data2 = np.array([-1, -1, -1, 1])
    #
    # nn_and = perceptron()
    # nn_and.train(data1, data2)
    # for test in data1:
    #     print test, '--->',nn_and.predict(test)

    #w1,w2
    data1 = np.array([[0.1,1.1],[6.8,7.1],[-3.5,-4.1],
                      [2.0,2.7],[4.1,2.8],[3.1,5.0],
                      [-0.8,-1.3],[0.9,1.2],[5.0,6.4],[3.9,4.0],
                     [7.1,4.2],[-1.4,-4.3],[4.5,0.0],
                      [6.3,1.6],[4.2,1.9],[1.4,-3.2],
                      [2.4,-4.0],[2.5,-6.1],[8.4,3.7],[4.1,-2.2]])
    data2 = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1])
    nn_12 = perceptron()
    print 'epochs',nn_12.train(data1,data2)
    print 'weight',nn_12.weight
    for test in data1:
        r = nn_12.predict(test)
        if r > 0:
            result = 'w2'
        else:
            result = 'w1'
        print test, '--->', result

    #w3,w4
    data3 = np.array([[-3.0,-2.9],[0.5,8.7],[2.9,2.1],
                      [-0.1,5.2],[-4.0,2.2],[-1.3,3.7],
                      [-3.4,6.2],[-4.1,3.4],[-5.1,1.6],[1.9,5.1],
                      [-2.0,-8.4],[-8.9,0.2],[-4.2,-7.7],
                      [-8.5,-3.2],[-6.7,-4.0],[-0.5,-9.2],
                      [-5.3,-6.7],[-8.7,-6.4],[-7.1,-9.7],[-8.0,-6.3]])
    data4 = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    nn_34 = perceptron()
    print 'epochs',nn_34.train(data3, data4)
    print 'weight',nn_34.weight
    for test in data3:
        r= nn_34.predict(test)
        if r > 0:
            result = 'w4'
        else:
            result = 'w3'
        print test, '--->', result

    draw(data1,data3)














            