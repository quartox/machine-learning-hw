"""Identifies how often the one vs all logistic regression gets the solution."""

__author__="Jesse Lord"
__date__="January 9, 2015"

import numpy as np
from sigmoid import sigmoid

def predictOneVsAll(X,y,all_thetas,num_labels):
    y = y[:,0]
    ndims = X.shape
    n = ndims[0]
    m = ndims[1]
    newX = np.ones([n,m+1])
    newX[:,1:] = X
    for ii in range(num_labels):
        h = sigmoid(np.dot(newX,all_thetas[ii,:]))
        if ii==0:
            hmax = h
            itermax = np.zeros([len(h)])+10
        else:
            for jj in range(len(h)):
                if h[jj] > hmax[jj]:
                    hmax[jj] = h[jj]
                    itermax[jj] = ii
                if ii == num_labels-1:
                    if jj == 0:
                        prediction = 0.0
                    if itermax[jj] == y[jj]:
                        prediction += 1.0/n
                # endif h[jj] > hmax[jj]
            # endfor jj in range(len(h))
        # endif ii==0
    # endfor ii in range(num_labels)

    return prediction
