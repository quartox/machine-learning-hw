"""Computes the sigmoid function and gradient."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z)*(1.0-sigmoid(z))
