"""Uses scikit-learn support vector machine software to classify our data sets."""

__author__="Jesse Lord"
__date__="January 19, 2015"

from sklearn import svm
import numpy as np
from plotGauss import plotGauss

def linearKernel(X,y,C):
    """Computes the classification boundary using a linear kernel."""
    num_features = X.shape
    num_features = num_features[1]
    y = np.ravel(y)
    clf = svm.LinearSVC(C=C)
    clf.fit(X,y)
    theta = np.empty(num_features+1)
    theta[0] = clf.intercept_
    theta[1:] = clf.coef_
    return theta

def gaussianKernel(X,y,C,sigma):
    """Computes the classification boundary using a gaussian kernel."""
    y = np.ravel(y)
    # To compute the gaussian kernel use the gamma function with
    # gamma = 1 / (2 sigma^2)
    gamma = 1.0 / (2.0 * sigma*sigma)
    clf = svm.SVC(C=C,kernel='rbf',gamma=gamma)
    clf.fit(X,y)
    plotGauss(X,y,clf)
    return clf

def validation(X,y,Xcv,ycv):
    """Determines the best choice of regularization and gaussian parameters on cross validation set."""
    y = np.ravel(y)
    ycv = np.ravel(ycv)
    # loop through a range of regularization and gaussian parameters
    # from 0.01 to ~30 increasing each value by a factor of 3
    bestParams = np.zeros(3)
    C = 0.01
    sigma = 0.01
    while C < 32:
        while sigma < 32:
            # To compute the gaussian kernel use the gamma function with
            # gamma = 1 / (2 sigma^2)
            gamma = 1.0 / (2.0 * sigma*sigma)
            clf = svm.SVC(C=C,kernel='rbf',gamma=gamma)
            clf.fit(X,y)
            # determining how well this fit predicts the cross validation data
            pcv = clf.predict(Xcv)
            prediction = np.mean(pcv==ycv)
            if prediction > bestParams[0]:
                bestParams[0] = prediction
                bestParams[1] = C
                bestParams[2] = sigma
            # iterate the parameters by a factor of 3
            sigma *= 3.0
        # endwhile sigma < 32
        C *= 3.0
        sigma = 0.01 # reset sigma
    # endwhile C < 32

    # computing the fit with the best fit parameters to plot the
    # decision boundary
    C = bestParams[1]
    sigma = bestParams[2]
    gamma = 1.0 / (2.0 * sigma*sigma)
    clf = svm.SVC(C=C,kernel='rbf',gamma=gamma)
    clf.fit(X,y)
    plotGauss(X,y,clf)
    return bestParams
