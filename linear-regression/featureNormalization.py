"""Normalize the training set data for the features."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import numpy as np

def featureNormalization(X):
    X_norm = X
    shape = X.shape
    mu = np.empty(shape[0]-1)
    sigma = np.empty(shape[0]-1)
    for ii in range(1,shape[0]):
        mu[ii-1]=np.mean(X[ii,:])
        sigma[ii-1]=np.std(X[ii,:])
        X_norm[ii,:] = (X[ii,:]-mu[ii-1])/sigma[ii-1]
    return (X_norm,mu,sigma)
