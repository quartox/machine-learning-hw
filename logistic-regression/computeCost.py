"""Computes the cost function for an arbitrary number of features."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import numpy as np
from sigmoid import sigmoid

def computeCost(theta,X,y):
    m=len(y)
    z = np.dot(theta,X)
    h = sigmoid(z)
    J = sum((-y*np.log(h))-((1-y)*np.log(1-h)))/m
    return J

def computeRegularizedCost(theta,X,y,lambda):
    m=len(y)
    z = np.dot(theta,X)
    h = sigmoid(z)
    J = ( (lambda*sum(theta[1:]*theta[1:])/2.0) + sum((-y*np.log(h))-((1-y)*np.log(1-h))) )/m
    return J
