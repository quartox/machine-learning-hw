"""Computes the learning curve for linear regression."""

__author__="Jesse Lord"
__date__="January 14, 2015"

import readData
import costFunction

debug = 0

from readData import readData
from initialize import initialize
from regression import regression
from learningCurve import learningCurve
from validationCurve import validationCurve
from polyFeatures import polyFeatures
from featureNormalization import featureNormalization
import plotData
import numpy as np

if __name__=="__main__":

    (X,y,Xcv,ycv,Xtest,ytest) = readData()

    lam = 0 # regularization parameter
    degree = 1 # degree of the polynomial
    theta = initialize(degree)

    # checking code by printing values of cost, derivatives and plotting
    # the linear regression
    # we expect 303.951 for cost
    # then [ -15.30, 598.250 ] for gradient
    print "Cost for initial theta = [1,1]: ", \
        costFunction.computeCost(theta,X,y,lam)
    print "Gradient for initial theta = [1,1]: ", \
        costFunction.computeDeriv(theta,X,y,lam)
    theta = regression(theta,X,y,lam)
    plotData.plotTheta(X,y,theta)

    # Computing the learning curve for linear regression
    (error_train,error_cv) = learningCurve(theta,X,y,Xcv,ycv,lam)
    plotData.plotError(error_train,error_cv)

    # Unregularized polynomial regression for a degree 8 polynomial
    degree = 8
    theta = initialize(degree)
    X_poly = polyFeatures(X,degree)
    (X_norm,mu,sigma) = featureNormalization(X_poly)
    theta = regression(theta,X_norm,y,lam)
    #plotData.plotNorm(X,y,theta,mu,sigma)
    # Learning curve for a degree 8 polynomial
    Xcv_norm = polyFeatures(Xcv,degree)
    for ii in range(degree):
        Xcv_norm[:,ii] -= mu[ii]
        Xcv_norm[:,ii] /= sigma[ii]
    (error_train,error_cv) = learningCurve(theta,X_norm,y,Xcv_norm,ycv,lam)
    plotData.plotError(error_train,error_cv)

    # Computing the validation curve for different regularization values
    degree = 8
    theta = initialize(degree)
    (error_train,error_cv,lam) = validationCurve(theta,X_norm,y,Xcv_norm,ycv)
    plotData.plotError(error_train,error_cv,x=lam,
                       xlabel='Regularization parameter')

    # Computing the test error for a degree 8 polynomial with
    # a regularization parameter optimized on the cross-validation set
    lam = 2
    degree = 8
    theta = initialize(degree)
    theta = regression(theta,X_norm,y,lam)
    Xtest_norm = polyFeatures(Xtest,degree)
    for ii in range(degree):
        Xtest_norm[:,ii] -= mu[ii]
        Xtest_norm[:,ii] /= sigma[ii]
    print "The square error for the test set: ",costFunction.computeCost(theta,Xtest_norm,ytest,0)
