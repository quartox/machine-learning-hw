"""Computes the backward propagation to determine neural network weights."""

__author__="Jesse Lord"
__date__="January 13, 2015"

import costFunction
from sigmoid import sigmoidGradient
import numpy as np

def backPropagation(Theta1,Theta2,X,y,num_labels):

    num_examples = len(y)

    # start with forward propagation
    (a3,z2,a2,a1) = costFunction.computeH(Theta1,Theta2,X)
    Y = costFunction.matrixY(y,num_labels)

    # computing the error in the cost, i.e. the partial derivative of the cost
    delta3 = (a3-Y)
    delta2 = np.dot(np.transpose(Theta2[:,1:]),delta3)*sigmoidGradient(z2)
    Delta1 = np.dot(delta2,np.transpose(a1))/num_examples
    Delta2 = np.dot(delta3,np.transpose(a2))/num_examples
