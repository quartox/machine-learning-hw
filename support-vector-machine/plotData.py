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
