"""Computes the cost function and derivatives for linear regression."""

__author__="Jesse Lord"
__date__="January 14, 2015"

import numpy as np

def computeCost(theta,X,y,lam):
    y = y[:,0]
    ndims = X.shape
    num_training = ndims[0]
    num_features = ndims[1]
    a1 = np.ones([num_training,num_features+1])
    a1[:,1:] = X
    square = np.dot(a1,theta)-y
    J = ( np.sum(square*square) + \
          lam*np.sum(theta[1:]*theta[1:]) ) / (2.0*num_training)
    return J

def computeDeriv(theta,X,y,lam):
    y = y[:,0]
    ndims = X.shape
    num_training = ndims[0]
    num_features = ndims[1]
    a1 = np.ones([num_training,num_features+1])
    a1[:,1:] = X
    deriv = np.empty(num_features+1)
    for ii in range(num_features+1):
        deriv[ii] = np.sum(a1[:,ii]*(np.dot(a1,theta) - y))/num_training
        if ii > 0:
            deriv[ii] += lam*theta[ii]/num_training
    return deriv
