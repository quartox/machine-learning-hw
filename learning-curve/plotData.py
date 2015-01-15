"""Plots the scatter plot of the logistic regression training set."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import matplotlib.pyplot as p
import numpy as np
from mapFeature import mapFeature
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

def plotLearning(error_train,error_cv):
    n = len(error_train)
    x = np.arange(n)+1
    p.plot(x,error_train,'g')
    p.plot(x,error_cv,'b')

    p.show()

def plotReg(x,y,theta,degree):
    pos = np.where(y==1)
    neg = np.where(y==0)
    p.scatter(x[0,pos],x[1,pos],marker='o')
    p.scatter(x[0,neg],x[1,neg],marker='x')
    nx = 100
    rangex = np.array([np.amin(x[0,:]),np.amax(x[0,:])])
    dx = (rangex[1]-rangex[0])/nx
    ny = 100
    rangey = np.array([np.amin(x[1,:]),np.amax(x[1,:])])
    dy = (rangey[1]-rangey[0])/ny
    xplot = np.empty([2,1])
    xcontour = np.empty(nx)
    ycontour = np.empty(ny)
    zcontour = np.empty([nx,ny])
    for ii in range(nx):
        for jj in range(ny):
            xplot[0,0] = rangex[0] + ii*dx
            xcontour[ii] = xplot[0,0]
            xplot[1,0] = rangey[0] + jj*dy
            ycontour[jj] = xplot[1,0]
            (zplot,nfeatures) = mapFeature(xplot,degree)
            zcontour[ii,jj] = np.dot(zplot[:,0],theta)
        # endfor jj in range(ny)
    # endfor ii in range(nx)
    p.contour(xcontour,ycontour,zcontour,levels=[0])
    p.show()
