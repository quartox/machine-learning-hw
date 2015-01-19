"""Computes the best fit classification boundary between the two classes using a gaussian kernel."""

__author__="Jesse Lord"
__date__="January 19, 2015"

from gaussian import gaussianKernel
from readData import readData2
import numpy as np
import plotData
import supportVectorMachine

if __name__=="__main__":

    x1 = np.zeros(3)+1
    x1[1] += 1
    x2 = np.zeros(3)
    x2[1] = 4
    x2[2] = -1
    sigma = 2
    print "Evaluating the Gaussian Kernel (should be 0.324652): ", \
        gaussianKernel(x1,x2,sigma)

    (X,y) = readData2()

    plotData.plotPoints(X,y)

    C = 1.0
    sigma = 0.1

    supportVectorMachine.gaussianKernel(X,y,C,sigma)
