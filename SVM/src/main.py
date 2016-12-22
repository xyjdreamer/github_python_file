#-*- coding: UTF-8 -*-
from SMO import smop
from SMO import kernelTrans
import numpy

def testSMO():
    kernel = 'lin'
    k1 = 1.3
    dataArr = [[1,1],[0,0],[1,0],[0,1]]
    lableArr = [1,1,-1,-1]

    b,alpha = smop(dataArr, lableArr, 0.6, 0.001, 400, (kernel, k1))

    dataMat = numpy.mat(dataArr)
    lableMat = numpy.mat(lableArr).transpose()
    svInd = numpy.nonzero(alpha.A > 0)[0]
    sVs = dataMat[svInd]
    lableSV = lableMat[svInd]
    print "There are %d support vector" % numpy.shape(sVs)[0]

    m = numpy.shape(dataMat)[0]
    errorCount = 0
    for i in range(m):
        kernlEval = kernelTrans(sVs, dataMat[i,:],(kernel, k1))
        predict = kernlEval.T * numpy.multiply(lableSV, alpha[svInd]) + b
        if numpy.sign(predict) != numpy.sign(lableArr[i]):
            errorCount += 1
    print 'total error:',errorCount,"\tThe train error rate is %.2f" % (1.0*errorCount/m)

    dataArr1 = [[0,1],[1,0],[1,1],[0,0]]
    lableArr1 = [-1,-1,1,1]
    errorCount = 0
    dataMat = numpy.mat(dataArr1)
    lableMat = numpy.mat(lableArr1).transpose()
    m = numpy.shape(dataMat)[0]
    for i in range(m):
        kernlEval = kernelTrans(sVs, dataMat[i,:],(kernel, k1))
        predict = kernlEval.T * numpy.multiply(lableSV, alpha[svInd]) + b
        if numpy.sign(predict) != numpy.sign(lableArr1[i]):
            errorCount += 1
            print dataMat[i,:],'\tpredict value:',numpy.sign(predict),'\ttarget value:',lableArr1[i]
    print 'total error:',errorCount,"\tThe test error rate is %.2f" % (1.0*errorCount/m)
    
if __name__ == '__main__':
    testSMO()
