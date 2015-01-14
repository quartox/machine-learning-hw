"""Computes the backward propagation to determine neural network weights."""

__author__="Jesse Lord"
__date__="January 13, 2015"

import costFunction
import numpy as np
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b

def backPropagation(Theta1,Theta2,X,y,input_layer,hidden_layer,num_labels,lam):

    # Flattening the Thetas
    Thetas = np.reshape(Theta1,Theta1.size)
    Thetas = np.append(Thetas,Theta2)

    print "Starting l_bfgs_b minimization."

    (Thetas,f,d) = fmin_l_bfgs_b(costFunction.computeRegularizedCost,Thetas,
                                 fprime=costFunction.computeRegularizedDeriv,
                                 args=(X,y,input_layer,hidden_layer,num_labels,
                                       lam),maxiter=100,maxfun=100)

    print "Finished minimization with cost equal to: ",f

    (Theta1,Theta2) = costFunction.reshapeThetas(Thetas,input_layer,
                                                 hidden_layer,num_labels)

    return (Theta1,Theta2)
