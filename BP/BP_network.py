#!usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np

def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1.0 - tanh(x)**2

class NeuronNetwork():
    def __init__(self,layer,activation='tanh'):
        self.layer = layer
        if 'tanh' == activation:
            self.activation = tanh
            self.activation_prime = tanh_prime

        #initialize weight
        self.weight = []

        for i in range(1,len(self.layer)-1):
            self.weight.append(np.random.random((self.layer[i-1]+1,self.layer[i]+1)))  #add bais
        self.weight.append(np.random.random((self.layer[-2]+1,self.layer[-1])))

    def train(self,input,output,epochs=5000,learning_rate=0.2):
        input = np.hstack( ( input, np.ones((input.shape[0],1)) ) )
        iter = 0
        while iter < epochs:
            iter += 1
            index = np.random.randint(low=input.shape[0],high=None)
            a = [ input[index] ]
            #forward
            for i in range(len(self.weight)):
                dot_value = np.dot(a[-1],self.weight[i])
                activation = self.activation(dot_value)
                a.append(activation)
            #back
            error = output[index] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]
            for i in range(len(self.weight)-1,0,-1):
                deltas.append(np.dot(deltas[-1],self.weight[i].T)*self.activation_prime(a[i]))
            deltas.reverse()
            #update
            for i in range(len(self.weight)):
                y = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weight[i] += learning_rate * np.dot(y.T,delta)

    def predict(self,X):
        a = np.hstack((X,np.ones(1)))
        for i in range(len(self.weight)):
            a = self.activation(np.dot(a,self.weight[i]))
        return a


if __name__ == '__main__':
    nn = NeuronNetwork([2,2,1])
    input = np.array([[0,0],[0,1],[1,0],[1,1]])
    output = np.array([0,1,1,0])
    nn.train(input,output)
    print '1-th layer weight:\n',nn.weight[0]
    print '2-th layer weight:\n', nn.weight[1]
    print 'predict result:'
    for test in input:
        print test,nn.predict(test)



