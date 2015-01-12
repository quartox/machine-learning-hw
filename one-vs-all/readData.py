"""Reads in the data from the matlab files."""

__author__="Jesse Lord"
__date__="January 9, 2015"
import scipy.io as io

def readData():
    mat = io.loadmat("ex3data1.mat")
    y = mat['y']
    X = mat['X']
    return (X,y)
