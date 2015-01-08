"""Computes the gradient descent for linear regression with multiple features."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import numpy as np
from computeCost import computeCost
import matplotlib.pyplot as p

# turn on to compute cost function for each iteration as a debug
plotJ = 0

def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    nfeatures = len(theta)
    J_history = np.zeros(num_iters)
    for ii in range(num_iters):
        for jj in range(nfeatures):
            theta[jj] -= alpha*sum((np.dot(theta,X)-y)*X[jj,:])/m
        if plotJ:
            J_history[ii] = computeCost(X,y,theta)
    if plotJ:
        p.plot(J_history)
        p.show()
    return theta
