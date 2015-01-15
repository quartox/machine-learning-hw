"""Adds polynomial terms to the data set."""

__author__="Jesse Lord"
__date__="January 14, 2015"

import numpy as np

def polyFeatures(X,degree):
    num_features = X.size
    X_poly = np.empty([num_features,degree])
    X_poly[:,0] = X[:,0]
    for ii in range(2,degree+1):
        X_poly[:,ii-1] = X[:,0]**ii
    return X_poly
