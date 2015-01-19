"""Uses svm with a linear kernel to compute the best fit to classify the data."""

__author__="Jesse Lord"
__date__="January 19, 2015"

from readData import readData1
import numpy as np
import plotData
import supportVectorMachine

if __name__=="__main__":

    (X,y) = readData1()

    plotData.plotPoints(X,y)

    C = 1.0 # the regularization parameter

    theta = supportVectorMachine.linearKernel(X,y,C)

    plotData.plotTheta(X,y,theta)
