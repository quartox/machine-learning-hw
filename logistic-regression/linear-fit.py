
__author__="Jesse Lord"
__date__="January 8, 2015"

import numpy as np
import readData
import plotData
from computeCost import computeCost
from gradientDescent import logisiticDeriv
from scipy.optimize import fmin_bfgs

if __name__=="__main__":
    (x,y,nexamples) = readData.readFirst()

    plotData.plotPoints(x,y)

    nfeatures = x.shape
    nfeatures = nfeatures[0]
    X = np.ones([nfeatures+1,nexamples])
    X[1:,:] = x[:,:]

    theta = np.zeros(nfeatures+1)

    # should return 0.693 for the first data set
    print computeCost(theta,X,y)

    #iterations = 100000
    #alpha = 0.001
    #gradientDescent.gradientDescent(X,y,theta,alpha,iterations)
    theta=fmin_bfgs(computeCost,theta,fprime=logisiticDeriv,args=(X,y))

    plotData.plotTheta(x,y,theta)
