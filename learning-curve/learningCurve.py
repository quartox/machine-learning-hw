"""Computes linear regression for varying numbers of training examples."""

__author__="Jesse Lord"
__date__="January 14, 2015"

import numpy as np
from regression import regression
from costFunction import computeCost

def learningCurve(theta,X,y,Xcv,ycv,lam):
    num_training = y.size
    error_train = np.empty(num_training-1)
    error_cv = np.empty(num_training-1)
    # compute the error for varying number of training examples
    for ii in range(1,num_training):
        Xsubset = X[0:ii,:]
        ysubset = y[0:ii,:]
        theta_subset = regression(theta,Xsubset,ysubset,lam)
        # set lambda=0 to compute error in training and cross-validation sets
        error_train[ii-1] = computeCost(theta_subset,Xsubset,ysubset,0)
        error_cv[ii-1] = computeCost(theta_subset,Xcv,ycv,0)
    # endfor ii in range(1,num_training)
    return (error_train,error_cv)
