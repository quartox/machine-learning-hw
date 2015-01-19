"""Plots the scatter plot of the logistic regression training set."""

__author__="Jesse Lord"
__date__="January 19, 2015"

import matplotlib.pyplot as p
import numpy as np

def plotPoints(x,y):
    """Plots the scattered points with different markers for the two classes (y=1 or y=0)."""
    pos = np.where(y==1)
    neg = np.where(y==0)
    p.scatter(x[pos,0],x[pos,1],marker='o')
    p.scatter(x[neg,0],x[neg,1],marker='x')
    p.show()

def plotTheta(x,y,theta):
    """Plots the linear fit (i.e. x1*theta[1] + x2*theta[2] + theta[0] = 0) to the scattered points."""
    pos = np.where(y==1)
    neg = np.where(y==0)
    p.scatter(x[pos,0],x[pos,1],marker='o')
    p.scatter(x[neg,0],x[neg,1],marker='x')
    plotx = np.array([np.amin(x[:,0]),np.amax(x[:,0])])
    # plots the linear fit x1*theta[1] + x2*theta[2] + theta[0] = 0
    # by solving for x2
    p.plot(plotx,-(theta[0]+plotx*theta[1])/theta[2])
    p.show()
