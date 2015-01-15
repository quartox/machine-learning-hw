"""Initializes the features for linear regression."""

__author__="Jesse Lord"
__date__="January 14, 2015"

import numpy as np

def initialize(degree):
    # simple initialize to ones for now, could change if necessary
    theta = np.ones(degree+1)
    return theta
