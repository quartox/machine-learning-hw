"""Normalize the training set data for multiple."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import numpy as np

def featureNormalization(X):
    X_norm = X
    shape = X.shape
    mu = np.empty(shape[1])
    sigma = np.empty(shape[1])
    for ii in range(shape[1]):
        mu[ii]=np.mean(X[:,ii])
        sigma[ii]=np.std(X[:,ii])
        X_norm[:,ii] = (X[:,ii]-mu[ii])/sigma[ii]
    return (X_norm,mu,sigma)
