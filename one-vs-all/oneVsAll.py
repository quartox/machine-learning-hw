"""Computes logistic regression as one vs all for a specified number of labels."""

__author__="Jesse Lord"
__date__="January 9, 2015"

import numpy as np
import costFunction
from scipy.optimize import fmin_bfgs

def oneVsAll(X,y,num_labels,lam):
    ndims = X.shape
    m = ndims[0]
    n = ndims[1]

    # the matrix of theta values for each label and each pixel
    all_thetas = np.zeros([num_labels,n+1])

    initial_theta = np.zeros(n+1)

    newX = np.ones([m,n+1])
    newX[:,1:] = X[:,:]

    # re-organizing the y array to a one-dimensional array
    y = y[:,0]

    for ii in range(num_labels):
        # initializing the y array to all zeros
        newy = np.zeros(y.size)
        # finding the indices that are the current digit
        if ii==0:
            digit = np.where(y==10)
        else:
            digit = np.where(y==ii)
        newy[digit] = 1
        theta = fmin_bfgs(costFunction.computeRegularizedCost,initial_theta,
                          fprime=costFunction.regularizedLogisiticDeriv,
                          args=(newX,newy,lam))
        all_thetas[ii,:] = theta[:]
    # endfor ii in range(num_labels)

    return all_thetas
