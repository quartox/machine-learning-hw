"""Plots the scatter plot of the logistic regression training set."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import matplotlib.pyplot as p

def plotData(x,y):
    p.scatter(x[0,:],x[1,:],c=y)
    p.show()
