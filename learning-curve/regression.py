"""Linear regression using the bfgs function."""

__author__="Jesse Lord"
__date__="January 14, 2015"

import costFunction
from scipy.optimize import fmin_l_bfgs_b

def regression(theta,X,y,lam):

    (theta,f,d) = fmin_l_bfgs_b(costFunction.computeCost,theta,
                                 fprime=costFunction.computeDeriv,
                                 args=(X,y,lam))

    #print "Finished minimization with cost equal to: ",f

    return theta
