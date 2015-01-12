"""Computes the cost function and derivatives for logistic regression."""

__author__="Jesse Lord"
__date__="January 9, 2015"

import numpy as np
from sigmoid import sigmoid

def computeRegularizedCost(theta,X,y,lam):
    m=len(y)
    z = np.dot(X,theta)
    h = sigmoid(z)
    J = ( (lam*sum(theta[1:]*theta[1:])/2.0) + \
          sum((-y*np.log(h))-((1-y)*np.log(1-h))) )/m
    #print J,np.amin(theta),np.amax(theta)
    return J

def regularizedLogisiticDeriv(theta,X,y,lam):
    m = len(y)
    nfeatures = len(theta)
    deriv = np.empty(nfeatures)
    z = np.dot(X,theta)
    h = sigmoid(z)
    deriv[0] = sum((h-y)*X[:,0])/m
    for jj in range(1,nfeatures):
        deriv[jj] = (sum((h-y)*X[:,jj]) + lam*theta[jj])/m
    #print np.amin(deriv),np.amax(deriv)
    return deriv
