"""Computes the gradient descent for logistic regression."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import numpy as np
from computeCost import computeCost
from computeCost import computeRegularizedCost
from sigmoid import sigmoid
import matplotlib.pyplot as p

# turn on to compute cost function for each iteration as a debug
plotJ = 0

def regularizedLogisiticDeriv(theta,X,y,lam):
    m = len(y)
    nfeatures = len(theta)
    deriv = np.empty(nfeatures)
    z = np.dot(theta,X)
    h = sigmoid(z)
    deriv[0] = sum((h-y)*X[0,:])/m
    for jj in range(1,nfeatures):
        deriv[jj] = (sum((h-y)*X[jj,:]) + lam*theta[jj])/m
    return deriv

def logisiticDeriv(theta,X,y):
    m = len(y)
    nfeatures = len(theta)
    deriv = np.empty(nfeatures)
    z = np.dot(theta,X)
    h = sigmoid(z)
    for jj in range(nfeatures):
        deriv[jj] = sum((h-y)*X[jj,:])/m
    return deriv

def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    nfeatures = len(theta)
    J_history = np.zeros(num_iters)
    for ii in range(num_iters):
        z = np.dot(theta,X)
        h = sigmoid(z)
        for jj in range(nfeatures):
            theta[jj] -= alpha*sum((h-y)*X[jj,:])/m
        if plotJ:
            J_history[ii] = computeCost(X,y,theta)
    if plotJ:
        p.plot(J_history)
        p.show()
    return theta
