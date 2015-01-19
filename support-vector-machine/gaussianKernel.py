"""Implements a Gaussian kernel."""

__author__="Jesse Lord"
__date__="January 19, 2015"

from numpy import exp

def gaussianKernel(x1,x2,sigma):
    sqauresum = sum((x1-x2)*(x1-x2))
    return exp(-sqauresum / (2.0*sigma*sigma))
