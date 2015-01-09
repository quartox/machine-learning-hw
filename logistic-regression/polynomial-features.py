"""Uses logistic regression to do a polynomial fit to the data."""

__author__="Jesse Lord"
__date__="January 9, 2015"

import numpy as np
import readData
import plotData
from computeCost import computeRegularizedCost
from gradientDescent import regularizedLogisiticDeriv
from mapFeature import mapFeature
from scipy.optimize import fmin_bfgs

if __name__=="__main__":
    (x,y,nexamples) = readData.readSecond()

    #plotData.plotPoints(x,y)

    degree = 6

    (X,nfeatures) = mapFeature(x,degree)

    theta = np.zeros(nfeatures+1)

    lam = 1

    # should return 0.693 for the second data set
    print computeRegularizedCost(theta,X,y,lam)

    # this code is only used for my gradient descent
    # which converges quite slowly compared to bfgs method
    #iterations = 100000
    #alpha = 0.001
    #gradientDescent.gradientDescent(X,y,theta,alpha,iterations)
    theta=fmin_bfgs(computeRegularizedCost,theta,fprime=regularizedLogisiticDeriv,args=(X,y,lam))

    plotData.plotReg(x,y,theta,degree)
