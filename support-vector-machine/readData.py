"""Reads in the data from the matlab files."""

__author__="Jesse Lord"
__date__="January 14, 2015"
import scipy.io as io

def readData1():
    mat = io.loadmat("ex6data1.mat")
    y = mat['y']
    X = mat['X']
    return (X,y)

def readData2():
    mat = io.loadmat("ex6data2.mat")
    y = mat['y']
    X = mat['X']
    return (X,y)

def readData3():
    mat = io.loadmat("ex6data3.mat")
    # training set data
    y = mat['y']
    X = mat['X']
    # cross validation data
    ycv = mat['yval']
    Xcv = mat['Xval']
    return (X,y,Xcv,ycv)
