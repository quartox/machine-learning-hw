"""Reads in the data from the matlab files."""

__author__="Jesse Lord"
__date__="January 14, 2015"
import scipy.io as io

def readData():
    mat = io.loadmat("ex5data1.mat")
    y = mat['y']
    X = mat['X']
    ycv = mat['yval']
    Xcv = mat['Xval']
    ytest = mat['ytest']
    Xtest = mat['Xtest']
    return (X,y,Xcv,ycv,Xtest,ytest)
