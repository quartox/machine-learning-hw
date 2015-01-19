"""Uses scikit-learn support vector machine software to classify our data sets."""

__author__="Jesse Lord"
__date__="January 19, 2015"

from sklearn import svm
import numpy as np

def linearKernel(X,y,C):
    num_features = X.shape
    num_features = num_features[1]
    y = np.ravel(y)
    clf = svm.LinearSVC(C=C)
    clf.fit(X,y)
    theta = np.empty(num_features+1)
    theta[0] = clf.intercept_
    theta[1:] = clf.coef_
    return theta
