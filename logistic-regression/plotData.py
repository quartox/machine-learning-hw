"""Plots the scatter plot of the logistic regression training set."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import matplotlib.pyplot as p
import numpy as np

def plotPoints(x,y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    p.scatter(x[0,pos],x[1,pos],marker='o')
    p.scatter(x[0,neg],x[1,neg],marker='x')
    p.show()

def plotTheta(x,y,theta):
    pos = np.where(y==1)
    neg = np.where(y==0)
    p.scatter(x[0,pos],x[1,pos],marker='o')
    p.scatter(x[0,neg],x[1,neg],marker='x')
    plotx = np.array([np.amin(x[0,:]),np.amax(x[0,:])])
    ploty = np.array([np.amin(x[1,:]),np.amax(x[1,:])])
    p.plot(ploty,-(theta[0]+ploty*theta[2])/theta[1])
    p.show()
