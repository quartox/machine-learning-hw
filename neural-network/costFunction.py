"""Computes the cost function for a neural network."""

__author__="Jesse Lord"
__date__="January 12, 2015"

import numpy as np
import sigmoid

def computeH(Theta1,Theta2,X):
    """Computes the output of the neural network."""
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
    a2[1:,:] = sigmoid.sigmoid(z2)

    # computing the output layer values using the Theta2 matrix
    z3 = np.dot(Theta2,a2)
    h = sigmoid.sigmoid(z3) # the output of the neural network
    return (h,z2,a2,a1)

def matrixY(y,num_labels):
    """Computes the matrix form of y. Each column is zero except for the index of the true number in this example of hand-written numbers."""
    y = y[:,0]
    num_examples = len(y)
    num_labels = int(np.amax(y)-np.amin(y))+1
    # note that the indeces of "y" are off by 1 since octave starts indeces at 1
    Y = np.zeros([num_labels,num_examples])
    for ii in range(num_examples):
        if y[ii]==10:
            Y[9,ii] = 1
        else:
            Y[y[ii]-1,ii] = 1
    return Y

def computeCost(Theta1,Theta2,X,y,num_labels):
    """Computes the cost function for the neural network."""
    (h,z2,a2,a1) = computeH(Theta1,Theta2,X)
    Y = matrixY(y,num_labels)

    num_examples = len(y)

    J = np.sum((-Y*np.log(h))-((1-Y)*np.log(1-h)))/num_examples

    return J

def computeRegularizedCost((Theta1,Theta2),X,y,num_labels,lam):
    """Computes the regularized cost function for the neural network."""
    (h,z2,a2,a1) = computeH(Theta1,Theta2,X)
    Y = matrixY(y,num_labels)

    num_examples = len(y)

    # computed regularized cost function
    # note that we do not include the bias unit,
    # i.e. Theta1[:,0] and Theta2[:,0] in the regularization
    J = ( np.sum((-Y*np.log(h))-((1-Y)*np.log(1-h))) + \
          ( 0.5*lam*(np.sum(Theta1[:,1:]*Theta1[:,1:]) + \
                     np.sum(Theta2[:,1:]*Theta2[:,1:])) ) ) / num_examples

    return J

def computeRegularizedDeriv((Theta1,Theta2),X,y,num_labels,lam):
    num_examples = len(y)

    # start with forward propagation
    (a3,z2,a2,a1) = computeH(Theta1,Theta2,X)
    Y = matrixY(y,num_labels)

    # computing the error in the cost, i.e. the partial derivative of the cost
    delta3 = (a3-Y)
    delta2 = np.dot(np.transpose(Theta2[:,1:]),delta3) * \
             sigmoid.sigmoidGradient(z2)
    Delta1 = np.dot(delta2,np.transpose(a1))/num_examples
    Delta1[:,1:] += lam*Theta1[:,1:]/num_examples
    Delta2 = np.dot(delta3,np.transpose(a2))/num_examples
    Delta2[:,1:] += lam*Theta2[:,1:]/num_examples
    return (Delta1,Delta2)
