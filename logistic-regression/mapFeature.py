"""Maps the two features to polynomials up to the specified degree."""

__author__="Jesse Lord"
__date__="January 9, 2015"

import numpy as np

degree = 6
nfeatures = 0
for ii in range(1,degree+1):
    nfeatures += ii+1

def mapFeature(x):
    nexamples = x.shape
    nexamples = nexamples[1]
    output = np.ones([nfeatures+1,nexamples])
    iter = 1
    for ii in range(1,degree+1):
        for jj in range(0,ii+1):
            output[iter,:] = (x[0,:]**(ii-jj)) * (x[1,:]**(jj))
            iter += 1
        # endfor jj in range(0,ii+1):
    # endfor ii in range(1,degree+1):
    return (output,nfeatures)
