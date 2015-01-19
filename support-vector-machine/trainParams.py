"""Explores the regularization and sigma parameters of the SVM with gaussian kernel."""

__author__="Jesse Lord"
__date__="January 19, 2015"

from readData import readData3
import numpy as np
import plotData
import supportVectorMachine

if __name__=="__main__":

    (X,y,Xcv,ycv) = readData3()

    #plotData.plotPoints(X,y)

    bestParams = supportVectorMachine.validation(X,y,Xcv,ycv)
    print "Best fit parameters (C and sigma) are: ",bestParams[1:]," with ",bestParams[0]*100,"% of the cross validation set correctly classified."
