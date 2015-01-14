"""Randomizes initial weights for the neural network in the interval from [-epsilon,epsilon)."""

__author__="Jesse Lord"
__date__="January 13, 2015"

import numpy as np

def randomInit(L_in,L_out):
    """Randomizes weights in range [-epsilon,epsilon). We choose epsilon=sqrt(6)/sqrt(L_in+L_out) where L_in is the number of input connections and L_out is the number of output connections in the neural network."""
    epsilon = np.sqrt(6.0) / np.sqrt( L_in + L_out )
    weights = np.random.rand(L_out,L_in+1)
    weights -= 0.5 # centers random numbers on 0.0
    weights *= 2.0*epsilon # changes range to [-epsilon,epsilon)
    return weights
