"""Computes the linear regression on a homework provided data set."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import numpy as np
import matplotlib.pyplot as plot
from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalization import featureNormalization
import readData

nfeatures = 2

if __name__=="__main__":
    if nfeatures == 1:
        (x,y,nexamples) = readData.readSingleFeature()
    elif nfeatures == 2:
        (x,y,nexamples) = readData.readMultiFeature()
    # transforming the X array into a matrix to simplify the
    # matrix multiplication with the theta_zero feature
    X = np.ones((nfeatures+1,nexamples))
    X[1:,:]=x[:,:]
    theta = np.zeros(nfeatures+1)

    if nfeatures==2:
        (X_norm,mu,sigma) = featureNormalization(X)

    # computes the cost as a test, should return 32.07
    print computeCost(X_norm,y,theta)

    if nfeatures == 1:
        iterations = 1500
    elif nfeatures == 2:
        iterations = 400
    alpha = 0.01

    # computes the linear regression coefficients using gradient descent
    theta = gradientDescent(X_norm,y,theta,alpha,iterations)

    print theta[0]+theta[1]*((1650-mu[0])/sigma[0])+theta[2]*((3-mu[1])/sigma[1])

    if nfeatures==1:
        plot.plot(x,y,'o',x,np.dot(theta,X))
        plot.show()
    #plot.plot(x[0,:],y,'o',x[0,:],np.dot(theta[:1],X[:1,:])
    #plot.show()
