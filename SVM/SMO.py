#!usr/bin/env python
#-*- coding: utf-8 -*-


import numpy as np
import random
class SVM:
    def __init__(self,x,y,tol,C,maxIter):
        self.x = np.mat(x)
        self.y = y
        self.tol = tol
        self.C = C
        self.b = 0
        self.m = np.size(x,0)   #计算训练样本个数
        self.a = np.zeros(self.m)    #初始化拉格朗日乘子
        self.maxIter = maxIter
        self.eCache = np.zeros((self.m, 2)) #用于存储位于边界上的点，即支持向量, 0 < alpha < C

    def k(self,predictX):
        return self.x * predictX.T  #带预测样例与各个训练样例的内积，返回值为矩阵类型

    def kernal(self,X1,X2):   #核函数,内积
        return np.inner(X1,X2)

    def f(self,i):
        return self.a * self.y * self.k(self.x[i]) + self.b

    def randomSelect(self,i):
        j = i
        while (j == i):
            j = int(random.uniform(0,self.m))
        return j

    def getE(self,i):
        return self.f(i) - float(self.y[i])

    #简化版本的SMO算法跳过了选择第二个alpha的过程，通过遍历alpha选择第一个不满足KKT条件的alpha，
    #再从剩余的alpha中随机选择一个alpha作为第二个alpha的值
    def SimplifiedSMO(self):
        iter = 0
        while(iter < self.maxIter):  #外循环
            alphaPairsChanged = 0  #表示乘子改变的次数
            # 内循环，依次检索每一个alpha
            for i in range(self.m):
                #判断Xi是否满足KKT条件
                Ei = self.getE(i)
                if (self.y[i]*Ei < -self.tol and self.a[i] < self.C)\
                    or (self.y[i]*Ei > self.tol and self.a[i] > 0) :

                    j = self.randomSelect(i) #随机选择第二个拉格朗日乘子

                    oldAi = self.a[i].copy()
                    oldAj = self.a[j].copy()

                    #计算上下界L、H
                    if self.y[i] != self.y[j]:
                        L = max(0, oldAj - oldAi)
                        H = min(self.C, self.C + oldAj - oldAi)
                    else:
                        L = max(0, oldAj + oldAi - self.C)
                        H = min(self.C, oldAj + oldAi)
                    if L == H : print 'L==H'; continue
                    #计算a1、a2更新后的值
                    Ej = self.getE(j)
                    eta = 2.0 * self.kernal(self.x[i],self.x[j]) - self.kernal(self.x[i],self.x[i]) -\
                          self.kernal(self.x[j],self.x[i])

                    #如果eta等于0或者大于0 则表明a最优值应该在L或者U上
                    if eta >= 0 :print 'eta>=0'; continue
                    newAj = oldAj - self.y[j] * (Ei - Ej) / eta
                    if newAj > H:
                        newAj = H
                    elif newAj < L:
                        newAj = L

                    self.a[j] = newAj
                    if (abs(oldAj - newAj) < 0.00001):
                        print "j not moving enough"
                        continue
                    self.a[i] = newAi = oldAi + self.y[i] * self.y[j] * (oldAj - newAj)

                    #计算b1、b2更新后的值
                    newb1 = self.b - Ei - self.y[i] * (newAi - oldAi) * self.kernal(self.x[i],self.x[i]) \
                            - self.y[j] * (newAj - oldAj) * self.kernal(self.x[i],self.x[j])
                    newb2 = self.b - Ej - self.y[i] * (newAi - oldAi) * self.kernal(self.x[i], self.x[j]) \
                            - self.y[j] * (newAj - oldAj) * self.kernal(self.x[j], self.x[j])
                    if self.a[i] > 0 and self.a[i] < self.C:
                        self.b = newb1
                    elif self.a[j] > 0 and self.a[j] < self.C:
                        self.b = newb2
                    else:
                        self.b = (newb1 + newb2)/2.0
                    alphaPairsChanged += 1
                    print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)

            if (alphaPairsChanged == 0):  #当没有乘子改变时,iter递增
                iter += 1
            else:
                iter = 0
    def selectJ(self,i,Ei):
        self.eCache[i] = [1,Ei]
        maxDeltaE = 0
        maxK = -1
        Ej = 0
        validEcacheList = np.nonzero(self.eCache[:, 0])[0]
        if(len(validEcacheList)>1):
            for k in validEcacheList:
                if k == i: continue
                Ek = self.getE(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
        else:
            j = self.randomSelect(i)  # 随机选择第二个拉格朗日乘子
            Ej = self.getE(j)
        return  maxK ,Ej

    def innerL(self,i):
        Ei = self.getE(i)
        # 判断Xi是否满足KKT条件
        if (self.y[i] * Ei < -self.tol and self.a[i] < self.C) \
                or (self.y[i] * Ei > self.tol and self.a[i] > 0):

            j, Ej = self.selectJ(i, Ei)  # 选择第二个拉格朗日乘子

            oldAi = self.a[i].copy()
            oldAj = self.a[j].copy()

            # 计算上下界L、H
            if self.y[i] != self.y[j]:
                L = max(0, oldAj - oldAi)
                H = min(self.C, self.C + oldAj - oldAi)
            else:
                L = max(0, oldAj + oldAi - self.C)
                H = min(self.C, oldAj + oldAi)
            if L == H: print 'L==H'; return 0
            # 计算a1、a2更新后的值
            eta = 2.0 * self.kernal(self.x[i], self.x[j]) - self.kernal(self.x[i], self.x[i]) - \
                  self.kernal(self.x[j], self.x[i])

            # 如果eta等于0或者大于0 则表明a最优值应该在L或者U上
            if eta >= 0: print 'eta>=0'; return 0
            newAj = oldAj - self.y[j] * (Ei - Ej) / eta
            if newAj > H:
                newAj = H
            elif newAj < L:
                newAj = L

            self.a[j] = newAj
            if (abs(oldAj - newAj) < 0.00001):
                print "j not moving enough"
                return 0
            self.a[i] = newAi = oldAi + self.y[i] * self.y[j] * (oldAj - newAj)

            if self.a[j] < self.C and self.a[j] > 0:
                self.eCache[j] = [1, self.getE(j)]
            if self.a[i] < self.C and self.a[i] > 0:
                self.eCache[i] = [1, self.getE(i)]

                # 计算b1、b2更新后的值
            newb1 = self.b - Ei - self.y[i] * (newAi - oldAi) * self.kernal(self.x[i], self.x[i]) \
                    - self.y[j] * (newAj - oldAj) * self.kernal(self.x[i], self.x[j])
            newb2 = self.b - Ej - self.y[i] * (newAi - oldAi) * self.kernal(self.x[i], self.x[j]) \
                    - self.y[j] * (newAj - oldAj) * self.kernal(self.x[j], self.x[j])
            if self.a[i] > 0 and self.a[i] < self.C:
                self.b = newb1
            elif self.a[j] > 0 and self.a[j] < self.C:
                self.b = newb2
            else:
                self.b = (newb1 + newb2) / 2.0
            return 1
        else: return 0
    # 完整版本的SMO算法
    def PlattSMO(self):
        iter = 0
        alphaPairsChanged = 0  # 表示乘子改变的次数
        entireSet = True
        while (iter < self.maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  # 外循环
            alphaPairsChanged = 0  # 表示乘子改变的次数
            if entireSet:
                # 内循环，依次检索每一个alpha
                for i in range(self.m):
                    alphaPairsChanged += self.innerL(i)
                    print "fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
                    iter += 1
            else:#依次检索每一边界上的点
                nonBoundIs = np.nonzero((self.a > 0) * (self.a < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i)
                    print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            if entireSet:
                entireSet = False  # toggle entire set loop
            elif (alphaPairsChanged == 0):
                entireSet = True
            print "iteration number: %d" % iter




def readDataFromTxt(filename):
    data = np.loadtxt(filename,delimiter='\t')
    featureArray = data[:,0:2]
    labelArray = data[:,-1]
    return featureArray , labelArray
x,y = readDataFromTxt('testSet.txt')

import time
t1 = time.clock()
testSVM = SVM(x,y,0.001,0.6,500)
testSVM.PlattSMO()
t2 = time.clock()
print testSVM.b
print testSVM.a[testSVM.a>0]
print 'time:',t2-t1,'s'







