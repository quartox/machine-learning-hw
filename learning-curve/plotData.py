"""Plots the scatter plot of the logistic regression training set."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import matplotlib.pyplot as p
import numpy as np
from polyFeatures import polyFeatures

def plotPoints(x,y):
    p.scatter(x,y,marker='x')
    p.show()

def plotTheta(X,y,theta):
    p.scatter(X,y,marker='x')
    plotx = np.array([np.amin(X),np.amax(X)])
    degree = theta.size-1
    ploty = np.zeros(2) + theta[0]
    for ii in range(1,degree+1):
        ploty += theta[ii] * (plotx**ii)
    p.plot(plotx,ploty)
    p.show()

def plotNorm(X,y,theta,mu,sigma):
    """Plots the data and best fit curve for polynomial features."""
    p.scatter(X[:,0],y[:,0],marker='x')
    # finding the range of x-values
    xstart = np.amin(X[:,0])
    xstop = np.amax(X[:,0])
    step = (xstop-xstart)/100.
    # extending the range of the x values
    xstart -= step*10
    xstop += step*10
    xplot = np.arange(xstart,xstop,step)
    # reshaping into a [100,1] array, since polyFeatures assumes this shape
    xplot = np.reshape(xplot,[xplot.size,1])
    degree = theta.size-1
    xpoly = polyFeatures(xplot,degree)
    for ii in range(degree):
        xpoly[:,ii] -= mu[ii]
        xpoly[:,ii] /= sigma[ii]
    # endfor
    # adding the bias unit for the dot product with theta
    ndims = xpoly.shape
    X = np.ones([ndims[0],ndims[1]+1])
    X[:,1:] = xpoly
    yplot = np.dot(theta,np.transpose(X))
    p.plot(xplot,yplot,'r')
    p.show()

def plotError(error_train,error_cv,x=None):
    """Plots the training error and cross validation error for a given range of parameters. If no x-axis is sent to function then it is assumed to be learning curve with number of training examples as x-axis."""
    if x is None:
        n = len(error_train)
        x = np.arange(n)+1
    p.plot(x,error_train,'g')
    p.plot(x,error_cv,'b')

    p.show()
