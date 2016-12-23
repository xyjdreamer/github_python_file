#-*- coding: UTF-8 -*-

from numpy import mat
from numpy import shape
from numpy import zeros
from numpy import multiply
from numpy import nonzero
from numpy import exp
from random import uniform
import matplotlib.pyplot as plt

class SMO(object):
    def __init__(self,dataMatIn, classLable, C, toler, kTup):
        self.X = dataMatIn
        self.lableMat = classLable
        self.C = C
        self.tol = toler #松弛变量
        self.m = shape(dataMatIn)[0]
        self.alpha = mat(zeros((self.m, 1))) 
        self.b = 0
        self.eCache = mat(zeros((self.m, 2))) 
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def kernelTrans(X, A, kTup):
    m = shape(X)[0]
    K = mat(zeros((m, 1)))
    p = 1
    if kTup[0] == 'lin':
        if p == 1:
            K = 1 + X * A.T
        else:
            temp = 1 + X * A.T
            for i in range(2,p+1):
                temp = multiply(temp,1 + X * A.T)
            K = temp
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K / (-2*kTup[1]**2))
    else:
        raise NameError('Houston We Have a problem -- That Kernel is not recognized')
    return K
         
def calEk(os, k):
    fXk = float(multiply(os.alpha,os.lableMat).T * os.K[:,k] + os.b)
    Ek = fXk - float(os.lableMat[k])
    return Ek

def selectJrand(i, m):
    j = i
    while j== i:
        j = int(uniform(0, m))
    return j

def selectJ(i, os, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = [1, Ei]
    validEcaheList = nonzero(os.eCache[:,0].A)[0]
    
    if len(validEcaheList) > 1:
        for k in validEcaheList:
            if k == i:
                continue
            Ek = calEk(os, k)
            deltaK = abs(Ei - Ek)
            if deltaK > maxDeltaE:
                maxDeltaE = deltaK
                maxK = k
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, os.m)
        Ej = calEk(os, j)
    return j, Ej

def updateEk(os, k):
    Ek = calEk(os, k)
    os.eCache[k] = [1, Ek]

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def innerL(i, os):
    Ei = calEk(os, i)
    
    if ((os.lableMat[i]*Ei < -os.tol and (os.alpha[i] < os.C)) or (os.lableMat[i]*Ei > os.tol and (os.alpha[i] > 0))):
        j,Ej = selectJ(i, os, Ei)
        alphaIold = os.alpha[i].copy()
        alphaJold = os.alpha[j].copy()
        if (os.lableMat[i] != os.lableMat[j]):
            L = max(0, os.alpha[j] - os.alpha[i])
            H = min(os.C, os.C + os.alpha[j] - os.alpha[i])
        else:
            L = max(0, os.alpha[j] + os.alpha[i] - os.C)
            H = min(os.C, os.alpha[j] + os.alpha[i])
        if L == H:
            print "L == H,"
            return 0
#         eta = 2.0 * os.X[i,:]*os.X[j,:].T - os.X[i,:]*os.X[i,:].T - os.X[j,:]*os.X[j,:].T
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j]
        if eta >= 0:print "eta>=0";return 0
        os.alpha[j] = os.alpha[j] - os.lableMat[j]*(Ei - Ej)/eta
        os.alpha[j] = clipAlpha(os.alpha[j], H, L)
        if (abs(os.alpha[j] - alphaJold) < 0.001):
            updateEk(os, j)
#             print "j not moving enough"
            return 0
        os.alpha[i] = os.alpha[i] + os.lableMat[j]*os.lableMat[i]*(alphaJold - os.alpha[j])
        updateEk(os, i)
#         b1 = os.b - Ei - os.lableMat[i]*(os.alpha[i] - alphaIold)*os.X[i, :]*os.X[i, :].T - os.lableMat[j]*(os.alpha[j] - alphaJold)*os.X[i, :]*os.X[j, :].T
        b1 = os.b - Ei - os.lableMat[i]*(os.alpha[i] - alphaIold)*os.K[i,i] - os.lableMat[j]*(os.alpha[j] - alphaJold)*os.K[i,j]
        b2 = os.b - Ej - os.lableMat[i]*(os.alpha[i] - alphaIold)*os.K[i,j] - os.lableMat[j]*(os.alpha[j] - alphaJold)*os.K[j,j]
        if (0 < os.alpha[i]) and (os.C > os.alpha[i]): 
            os.b = b1
        elif (0 < os.alpha[j]) and (os.C > os.alpha[j]): 
            os.b = b2
        else: os.b = (b1 + b2)/2.0
        return 1        
    else:
        return 0

def smop(dataMatIn, classLables, C, toler, maxIter, kTup=('lin',0)):
    os = SMO(mat(dataMatIn), mat(classLables).transpose(), C, toler, kTup)
    iteration = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iteration < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged = alphaPairsChanged + innerL(i, os)
            print "fullSet, iteration: %d i %d,pairs changed %d" % (iteration,i,alphaPairsChanged)
            iteration = iteration + 1
        else:
            nonBounds = nonzero((os.alpha.A > 0) * (os.alpha.A < C))[0]
            for i in nonBounds:
                alphaPairsChanged = alphaPairsChanged + innerL(i, os)
                print "non-bound, iteration: %d i %d,pairs changed %d" % (iteration,i,alphaPairsChanged)
            iteration = iteration + 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print "iteration number: %d" %iteration
    # showSVM(os)
    return os.b,os.alpha

def showSVM(os):
    if os.X.shape[1] != 2:
        print "Sorry,I can't draw because the dimension of your data is not 2!"
        return 1
    for i in xrange(os.m):
        if os.lableMat[i] == 0:
            plt.plot(os.X[i,0], os.X[i,1],'or')
        elif os.lableMat[i] == 1:
            plt.plot(os.X[i,0], os.X[i,1],'ob')
    supportVectorsIndex = nonzero(os.alpha.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(os.X[i,0],os.X[i,1], 'oy')
    
    w = zeros((2, 1))
    for i in supportVectorsIndex:
        w += multiply(os.alpha[i] * os.lableMat[i], os.lableMat[i, :].T)
    # min_x = min(os.lableMat[:,0])[0,0]
    # max_x = max(os.lableMat[:,0])[0,0]
    # y_min_x = float(-os.b - w[0]*min_x)
    # y_max_x = float(-os.b - w[0]*max_x)
    # plt.plot([min_x, max_x],[y_min_x,y_max_x],'-g')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    # plt.show()