"""Plots the gaussian kernel decision boundary for the data."""

__author__="Jesse Lord"
__date__="January 19, 2015"

import matplotlib.pyplot as p
import numpy as np
from sklearn import svm

def plotGauss(x,y,clf):

    # plots the scatter plot of the two classes of data with different markers
    pos = np.where(y==1)
    neg = np.where(y==0)
    p.scatter(x[pos,0],x[pos,1],marker='o')
    p.scatter(x[neg,0],x[neg,1],marker='x')

    # creates a range of values for a contour plot to
    # show the decision boundary
    nx = 200
    rangex = np.array([np.amin(x[:,0]),np.amax(x[:,0])])
    dx = (rangex[1]-rangex[0])/nx
    ny = 200
    rangey = np.array([np.amin(x[:,1]),np.amax(x[:,1])])
    dy = (rangey[1]-rangey[0])/ny
    xplot = np.empty([nx*ny,2])
    xcontour = np.empty(nx)
    ycontour = np.empty(ny)
    #zcontour = np.empty([nx,ny])
    # loops through the x1 and x2 values and uses the
    # svm.predict function to determine contours
    for ii in range(nx):
        for jj in range(ny):
            xplot[ii*nx+jj,0] = rangex[0] + ii*dx
            xcontour[ii] = xplot[ii*nx+jj,0]
            xplot[ii*nx+jj,1] = rangey[0] + jj*dy
            ycontour[jj] = xplot[ii*nx+jj,1]
        # endfor jj in range(ny)
    # endfor ii in range(nx)
    zcontour = clf.predict(xplot)
    zcontour = np.transpose(np.reshape(zcontour,[nx,ny]))
    p.contour(xcontour,ycontour,zcontour,levels=[0])
    p.show()
