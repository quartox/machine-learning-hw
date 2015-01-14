"""Numerically computes derivative and compares to analytic derivatives."""

__author__="Jesse Lord"
__date__="January 13, 2015"

import costFunction
import numpy as np

def gradientChecking(Theta1,Theta2,X,y,input_layer,hidden_layer,num_labels,lam):
    #(Theta1,Theta2) = costFunction.reshapeThetas(Thetas,input_layer,
    #                                             hidden_layer,num_labels
    # Flattening the derivatives
    Thetas = np.reshape(Theta1,Theta1.size)
    Thetas = np.append(Thetas,Theta2)

    Derivs = costFunction.computeRegularizedDeriv(Thetas,X,y,input_layer,
                                                  hidden_layer,num_labels,lam)
    epsilon = 1.0e-4
    plus = Thetas
    plus[0] += epsilon
    Cplus = costFunction.computeRegularizedCost(plus,X,y,input_layer,
                                                hidden_layer,num_labels,lam)
    minus = Thetas
    minus[0] -= epsilon
    Cminus = costFunction.computeRegularizedCost(minus,X,y,input_layer,
                                                 hidden_layer,num_labels,lam)
    f = (Cplus-Cminus)/(2.0*epsilon)
    print f,Derivs[0],Thetas[0]
    #print np.amin(numerical_deriv),np.amax(numerical_deriv)
    #print np.amin(Derivs),np.amax(Derivs)
    #print np.amax(abs(numerical_deriv-Derivs))
