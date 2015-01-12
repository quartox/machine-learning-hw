"""Computes the cost function for a neural network."""

__author__="Jesse Lord"
__date__="January 12, 2015"

import numpy as np
from sigmoid import sigmoid

def computeCost(Theta1,Theta2,X,y,num_labels):

    y = y[:,0]
    ndims = X.shape
    num_examples = ndims[0]
    num_pixels = ndims[1]
    a1 = np.ones([num_examples,num_pixels+1])
    a1[:,1:] = X
    a1 = np.transpose(a1)
    # computing the hidden layer values using the Theta1 matrix
    z2 = np.dot(Theta1,a1)
    # creating the hidden layer outputs using the sigmoid function
    ndims = z2.shape
    a2 = np.ones([ndims[0]+1,num_examples])
    a2[1:,:] = sigmoid(z2)

    # computing the output layer values using the Theta2 matrix
    z3 = np.dot(Theta2,a2)
    h = sigmoid(z3) # the output of the neural network

    # putting the known answers of the training set in the correct matrix form
    # note that the indeces of "y" are off by 1 since octave starts indeces at 1
    ycomp = np.zeros([num_labels,num_examples])
    for ii in range(num_examples):
        if y[ii]==10:
            ycomp[9,ii] = 1
        else:
            ycomp[y[ii]-1,ii] = 1

    J = np.sum((-ycomp*np.log(h))-((1-ycomp)*np.log(1-h)))/num_examples

    print J
